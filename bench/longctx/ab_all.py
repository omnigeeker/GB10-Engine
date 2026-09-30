#!/usr/bin/env python3
"""Same-session cold-TTFT A/B: gb10 (FA2 off / FA2 on) vs llama.cpp, at exact contexts.

This is the driver that produces the evidence the objective requires. It exists
because the previous comparisons were assembled by hand: one engine on one port,
servers started and killed manually, and the prompt length inferred from a `reps`
count rather than from the token count the server actually saw. That is enough to
notice a regression and not enough to *prove* a win.

What it does differently:

  * It calibrates `reps` against the server's own `prompt_tokens`, so a requested
    context is the context both engines actually receive. The calibration is a
    two-point linear fit (reps=8 and reps=64), because the filler-to-token ratio
    is not exactly linear -- the chat template adds a constant, and rounding is
    per-request.
  * It records the `prompt_tokens` each engine reported, so a comparison can show
    the two were handed the same prompt instead of asserting it.
  * It measures the *minimum* cold TTFT over `--trials`, because the observed
    drift on this box is large (23% between sessions) and the minimum is the
    least contaminated estimate of what the hardware can do.
  * It runs every configuration in ONE session, in sequence, with a health check
    and a hard kill between them, so no two GPU measurements overlap.

Usage:
  # full four-context proof (this is a multi-hour run -- use a background job)
  python3 bench/longctx/ab_all.py --contexts 8192,32768,131072,262144 \
      --out bench/longctx/ab-$(date +%Y%m%d-%H%M).md

  # quick check of one engine/context while iterating
  python3 bench/longctx/ab_all.py --contexts 32768 --engines gb10-fa2on --trials 1

Ordering note: the configurations run in the order given by --engines, and the
default order is gb10-fa2off, gb10-fa2on, llama. Because drift between *sessions*
is large but drift *within* a session is small, all three are kept in one session
and the run should not be interrupted and resumed piecemeal -- if it is, the
numbers stop being comparable and the whole point is lost.
"""
import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# Reuse the streaming measurement from ttft.py rather than reimplementing it, so
# the numbers here are directly comparable with the ones already in the record.
_spec = importlib.util.spec_from_file_location("ttft", os.path.join(HERE, "ttft.py"))
ttft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ttft)

GB10_PORT = 8080
LLAMA_PORT = 8899
GB10_MODEL = "models/Qwen3.8-27B-NVFP4"
LLAMA_MODEL = "models/Qwen3.8-27B-NVFP4.gguf"
CTX = 262144

# (name, extra env for the server process)
ENGINES = {
    "gb10-fa2off": {"kind": "gb10", "env": {"GB10_FA2": "0"}},
    "gb10-fa2on": {"kind": "gb10", "env": {"GB10_FA2": "1"}},
    "llama": {"kind": "llama", "env": {}},
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kill_servers():
    """Kill by exact name. Never `pkill -f`: the pattern would also match this
    script's own command line and the shell running it."""
    for name in ("gb10-server", "llama-server"):
        subprocess.run(["pkill", "-x", name], capture_output=True)
    time.sleep(3)


def start_server(engine):
    spec = ENGINES[engine]
    env = dict(os.environ)
    env.update(spec["env"])
    if spec["kind"] == "gb10":
        cmd = ["./target/release/gb10-server", "--model", GB10_MODEL,
               "--port", str(GB10_PORT), "--ctx", str(CTX)]
        port = GB10_PORT
    else:
        cmd = ["tools/llama.cpp/build/bin/llama-server", "-m", LLAMA_MODEL,
               "--port", str(LLAMA_PORT), "-c", str(CTX), "-ngl", "99", "-fa", "on"]
        port = LLAMA_PORT
    log(f"starting {engine}: {' '.join(cmd)}  env={spec['env']}")
    logf = open(os.path.join(HERE, f"server-{engine}.log"), "w")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    return proc, port


def wait_healthy(engine, port, timeout=900):
    """Poll until the server can serve.

    The two servers do not agree on what health looks like. gb10 answers
    /v1/models when it is ready. llama-server's /health returns HTTP 200 even
    while it is still loading, with a `Loading model` body and a 503 status
    inside the JSON -- so grepping the *body* for ok is the only reliable check.
    """
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if engine == "llama":
                r = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
                if r.status_code == 200 and '"ok"' in r.text:
                    return True
            else:
                r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                if r.status_code == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def calibrate(url, label):
    """Solve for filler reps that yield a target token count.

    Two points, because the relationship is affine, not proportional: the chat
    template contributes a constant, and the question adds a fixed tail.
    """
    pts = []
    for reps in (8, 64):
        marker = f"cal-{label}-{reps}-{int(time.time())}"
        prompt = ttft.build_prompt(marker, reps)
        _, _, _, pt, _ = ttft.stream_once(url, prompt, 1, 600)
        if pt is None:
            raise RuntimeError(f"{label}: server did not report prompt_tokens")
        pts.append((reps, pt))
        log(f"  calibrate {label}: reps={reps} -> prompt_tokens={pt}")
    (r0, t0), (r1, t1) = pts
    per_rep = (t1 - t0) / (r1 - r0)
    overhead = t0 - per_rep * r0
    return per_rep, overhead


def reps_for(target, per_rep, overhead):
    return max(1, int(round((target - overhead) / per_rep)))


def measure(url, label, target, per_rep, overhead, trials, max_tokens, timeout):
    """Return (min_cold_ttft, prompt_tokens_seen, n_trials)."""
    reps = reps_for(target, per_rep, overhead)
    best = None
    seen = None
    for i in range(trials):
        marker = f"{label}-t{target}-{i}-{int(time.time())}"
        prompt = ttft.build_prompt(marker, reps)
        cold, _, _, pt, _ = ttft.stream_once(url, prompt, max_tokens, timeout)
        seen = pt
        best = cold if best is None else min(best, cold)
        log(f"  {label} target={target} reps={reps} trial={i} "
            f"prompt_tokens={pt} cold_ttft={cold:.2f}s")
    return best, seen, reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="8192,32768,131072,262144",
                    help="comma-separated target prompt token counts")
    ap.add_argument("--engines", default="gb10-fa2off,gb10-fa2on,llama")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=172800)
    ap.add_argument("--out", default=None, help="markdown output path")
    args = ap.parse_args()

    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    for e in engines:
        if e not in ENGINES:
            sys.exit(f"unknown engine {e!r}; known: {list(ENGINES)}")

    results = {}   # (engine, context) -> (cold_ttft, prompt_tokens, reps)
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit(f"# Same-session cold-TTFT A/B")
    emit()
    emit(f"- started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    emit(f"- contexts {contexts}")
    emit(f"- engines {engines}")
    emit(f"- trials {args.trials} (minimum reported), max_tokens {args.max_tokens}")
    emit(f"- host {os.uname().nodename}")
    emit()

    for engine in engines:
        kill_servers()
        proc, port = start_server(engine)
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        try:
            if not wait_healthy(engine, port):
                log(f"{engine}: FAILED to become healthy; skipping")
                emit(f"**{engine}: FAILED to become healthy -- no numbers.**")
                emit()
                continue
            log(f"{engine}: healthy")
            per_rep, overhead = calibrate(url, engine)
            log(f"{engine}: {per_rep:.3f} tokens/rep, overhead {overhead:.1f}")
            for ctx in contexts:
                best, seen, reps = measure(url, engine, ctx, per_rep, overhead,
                                           args.trials, args.max_tokens, args.timeout)
                results[(engine, ctx)] = (best, seen, reps)
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            kill_servers()
            time.sleep(5)

    # ---- report -------------------------------------------------------------
    emit()
    emit("## Cold TTFT (s)")
    emit()
    hdr = "| context | " + " | ".join(engines) + " |"
    emit(hdr)
    emit("|---|" + "---|" * len(engines))
    for ctx in contexts:
        cells = []
        for e in engines:
            v = results.get((e, ctx))
            cells.append(f"{v[0]:.2f}" if v else "n/a")
        emit(f"| {ctx} | " + " | ".join(cells) + " |")

    emit()
    emit("## Ratio vs llama.cpp (< 1.00 means gb10 is faster)")
    emit()
    base = "llama" if "llama" in engines else engines[-1]
    emit(f"| context | " + " | ".join(e for e in engines if e != base) + " |")
    emit("|---|" + "---|" * (len(engines) - 1))
    for ctx in contexts:
        b = results.get((base, ctx))
        cells = []
        for e in engines:
            if e == base:
                continue
            v = results.get((e, ctx))
            cells.append(f"{v[0] / b[0]:.3f}" if (v and b) else "n/a")
        emit(f"| {ctx} | " + " | ".join(cells) + " |")

    emit()
    emit("## Prompt tokens actually seen (proof both engines got the same prompt)")
    emit()
    emit("| context | " + " | ".join(engines) + " |")
    emit("|---|" + "---|" * len(engines))
    for ctx in contexts:
        cells = []
        for e in engines:
            v = results.get((e, ctx))
            cells.append(str(v[1]) if v and v[1] is not None else "n/a")
        emit(f"| {ctx} | " + " | ".join(cells) + " |")

    emit()
    emit(f"- finished {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
