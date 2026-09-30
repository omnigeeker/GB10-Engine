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
import glob
import importlib.util
import json
import os
import re
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
    # Same-binary A/B arms for the shared-activation-cast hoist in
    # `weights.rs`. Both run the FA2 kernel; only the hoist differs, so a pair
    # taken in one session is a controlled comparison of that one change.
    "gb10-hoist0": {"kind": "gb10", "env": {"GB10_FA2": "1", "GB10_CAST_HOIST": "0"}},
    "gb10-hoist1": {"kind": "gb10", "env": {"GB10_FA2": "1", "GB10_CAST_HOIST": "1"}},
    # Same-binary A/B arms for the FA2 staging-loop address arithmetic.
    # NOTE: that A/B is done by swapping `elementwise.ptx` between runs of the
    # SAME binary (the PTX path is baked in at compile time, but the file itself
    # is read at process start), so it is not an `ENGINES` entry -- see
    # bench/longctx/fa2_stage_ab.sh.
    "llama": {"kind": "llama", "env": {}},
}


def warm_page_cache():
    """Read both model files so no engine pays first-touch cost on them.

    This is not optional. A previous proof's baseline arm was invalidated
    because it was the *first* engine started and read the 21.9 GB weight file
    cold: 12.57 s at 8K where warm it is 9.67 s, a 15-23% error, while the
    other two arms reproduced to ~1%. The ~48 GB of model files fit in this
    box's page cache (121 GB RAM), so reading them once makes every arm warm.
    """
    paths = []
    d = os.path.join(ROOT, GB10_MODEL)
    if os.path.isdir(d):
        for root, _, files in os.walk(d):
            paths.extend(os.path.join(root, f) for f in sorted(files))
    else:
        paths.append(d)
    paths.append(os.path.join(ROOT, LLAMA_MODEL))
    for p in paths:
        if not os.path.isfile(p):
            log(f"warm: {p} not found, skipped")
            continue
        n = 0
        with open(p, "rb") as fh:
            while True:
                b = fh.read(64 << 20)
                if not b:
                    break
                n += len(b)
        log(f"warmed {os.path.relpath(p, ROOT)} ({n / 2**30:.2f} GiB)")


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


# Processes that mean someone else owns the GPU. `ps`, never `pgrep -f`: a
# pattern in the searching command line self-matches and has produced phantom
# readings on this box.
BUSY = ("gb10-verify", "gb10-server", "llama-server", "test-backend-ops")


def foreign_gpu_procs(allow=()):
    """Processes that mean someone else owns the GPU.

    Zombies must be excluded. A killed server that its parent has not reaped
    shows up as `[gb10-server] <defunct>` -- the name matches, but a zombie
    holds no GPU resources and is not contention. This guard produced exactly
    that false positive on its first real run (it aborted the 128K hoist A/B
    after the hoist0 arm, on the defunct husk of hoist0's own server). `ps -o
    stat` gives `Z` for a zombie; the `<defunct>` marker in `args` is the
    belt-and-braces check.
    """
    out = subprocess.run(["ps", "-eo", "pid,stat,args", "--no-headers"],
                         capture_output=True, text=True).stdout
    me = os.getpid()
    hits = []
    for ln in out.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        pid, _, rest = ln.partition(" ")
        stat, _, rest = rest.strip().partition(" ")
        if not pid.isdigit() or int(pid) == me:
            continue
        if stat.startswith("Z") or "<defunct>" in rest or "grep" in rest:
            continue
        if any(a in rest for a in allow):
            continue
        if any(b in rest for b in BUSY):
            hits.append(f"{pid} [{stat}] {rest}")
    return hits


def require_exclusive_gpu(where, allow=()):
    """Abort rather than record a contended number.

    A number produced under contention is worse than no number: it looks like
    evidence and is not. This box has already had one A/B pair invalidated that
    way. Note that `kill_servers()` between engines is not sufficient -- a
    *foreign* process (another agent's benchmark) can appear at any time, so the
    check has to run around every measurement, not just at startup.

    `allow` names the server this run started, which is of course expected to be
    present.
    """
    hits = foreign_gpu_procs(allow)
    if hits:
        sys.exit(f"GPU CONTENDED {where}: refusing to measure. Foreign processes:\n  "
                 + "\n  ".join(hits))


def measure(url, label, target, per_rep, overhead, trials, max_tokens, timeout,
            allow=()):
    """Return (min_cold_ttft, prompt_tokens_seen, n_trials)."""
    reps = reps_for(target, per_rep, overhead)
    best = None
    seen = None
    for i in range(trials):
        require_exclusive_gpu(f"before {label} ctx={target} trial={i}", allow)
        marker = f"{label}-t{target}-{i}-{int(time.time())}"
        prompt = ttft.build_prompt(marker, reps)
        cold, _, _, pt, _ = ttft.stream_once(url, prompt, max_tokens, timeout)
        seen = pt
        best = cold if best is None else min(best, cold)
        log(f"  {label} target={target} reps={reps} trial={i} "
            f"prompt_tokens={pt} cold_ttft={cold:.2f}s")
        require_exclusive_gpu(f"after {label} ctx={target} trial={i}", allow)
    return best, seen, reps


def ptx_path():
    hits = glob.glob(os.path.join(ROOT,
                    "target/release/build/gb10-cuda-*/out/elementwise.ptx"))
    return hits[0] if len(hits) == 1 else None


def binary_freshness():
    """The stale-binary guard -- the trap that silently cost 27% of a proof.

    The host's dynamic shared-memory request (`GB10_FA2_BC` in ops.rs) and the
    compiled key-tile width (`FA2_BC` in kernels/elementwise.cu) are two halves
    of ONE knob. A `gb10-server` built before that default changed 32 -> 16
    requests 32 KB for a 16 KB tile, which silently drops occupancy from
    4 CTAs/SM to 3 and loses the entire BC=16 win -- while every correctness
    gate still passes. Nothing fails loudly, so the driver has to check it.

    Returns (evidence_lines, stale_problems).
    """
    import hashlib
    lines, stale = [], []
    srcs = [p for p in (os.path.join(ROOT, "crates/gb10-cuda/src/ops.rs"),
                        os.path.join(ROOT, "kernels/elementwise.cu"))
            if os.path.exists(p)]
    src_mt = max(os.path.getmtime(p) for p in srcs) if srcs else 0
    for name, rel in (("gb10-server", "target/release/gb10-server"),
                      ("gb10-verify", "target/release/gb10-verify")):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            stale.append(f"{name} is missing at {rel}")
            continue
        mt = os.path.getmtime(path)
        lines.append(
            f"- `{name}` built {time.strftime('%H:%M:%S', time.localtime(mt))}, "
            f"newest kernel source {time.strftime('%H:%M:%S', time.localtime(src_mt))}")
        if mt < src_mt:
            stale.append(
                f"{name} is OLDER than the newest kernel source "
                f"({os.path.basename(max(srcs, key=os.path.getmtime))}) -- "
                f"run `cargo build --release` (the WHOLE workspace, not just "
                f"gb10-verify: that is how this trap was sprung)")
    ptx = ptx_path()
    if ptx:
        lines.append("- elementwise.ptx sha256[:12] `" +
                     hashlib.sha256(open(ptx, "rb").read()).hexdigest()[:12] + "`")
    else:
        stale.append("could not locate exactly one elementwise.ptx under target/")
    return lines, stale


def occupancy_preflight():
    """Assert the FA2 kernel really gets 4 CTAs/SM on this binary.

    mtimes can lie -- a build that does not recompile the kernel still refreshes
    the binary. This runs the kernel once at 8K and reads the driver's REAL
    occupancy, which is the check that cannot be faked. It aborts only on an
    EXPLICIT bad reading; a missing marker returns a warning instead of a
    failure, because a false failure in an abort-by-default harness is its own
    trap here (the `<defunct>` zombie and the `--report-api-errors` episode).

    Returns (occ_line_or_None, problem_or_None).
    """
    env = dict(os.environ)
    env.update({"GB10_FA2": "1", "GB10_ATTN_OCCUPANCY": "1"})
    cmd = ["./target/release/gb10-verify", "prefill-shape", "--model", GB10_MODEL,
           "--limit", "8192", "--max-seq", "8192"]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=1800)
    m = re.search(r"\[occ\] attn_prefill_fa2:.*", r.stdout + r.stderr)
    if not m:
        return None, "no [occ] line in the 8K preflight (not treated as fatal)"
    line = m.group(0).strip()
    bad = []
    if "binding REGS" not in line:
        bad.append("binding is not REGS")
    mb = re.search(r"by_regs (\d+)", line)
    if not mb or int(mb.group(1)) < 4:
        bad.append("by_regs < 4")
    md = re.search(r"dynamic_smem (\d+)", line)
    if not md or int(md.group(1)) != 16384:
        bad.append("dynamic_smem != 16384 (host request does not match the 16 KB tile)")
    return line, ("; ".join(bad) if bad else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="8192,32768,131072,262144",
                    help="comma-separated target prompt token counts")
    ap.add_argument("--engines", default="gb10-fa2off,gb10-fa2on,llama")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=172800)
    ap.add_argument("--out", default=None, help="markdown output path")
    ap.add_argument("--no-warm", action="store_true",
                    help="skip the page-cache warm-up (only if you know the model "
                         "files are already resident)")
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

    # Guard the stale-binary trap BEFORE spending hours on the GPU: a
    # `gb10-server` older than the kernel source silently loses the BC=16
    # occupancy win with every gate still green.
    fresh_lines, stale = binary_freshness()
    emit("## binaries under test")
    emit()
    for l in fresh_lines:
        emit(l)
    emit()
    for l in fresh_lines:
        log(l.lstrip("- "))
    if stale:
        for l in stale:
            log("STALE: " + l)
        sys.exit("STALE BINARY -- refusing to measure (this trap costs 27% and "
                 "cannot fail loudly):\n  " + "\n  ".join(stale))

    # The real check: mtimes can lie, the driver's occupancy cannot.
    if any(ENGINES[e]["kind"] == "gb10" for e in engines):
        log("preflight: asserting the FA2 occupancy on the 8K shape")
        occ_line, occ_bad = occupancy_preflight()
        if occ_line:
            log("  " + occ_line)
            emit(f"- occupancy preflight (8K): `{occ_line}`")
            emit()
        if occ_bad and occ_line:
            log("OCCUPANCY PREFLIGHT FAILED: " + occ_bad)
            sys.exit("OCCUPANCY PREFLIGHT FAILED -- this binary is not the "
                     "4-CTA/SM BC=16 configuration, so every gb10 number would be "
                     f"wrong: {occ_bad}\n  {occ_line}")
        elif occ_bad:
            log("  WARNING: " + occ_bad)

    if not args.no_warm:
        log("warming page cache for both model files")
        warm_page_cache()

    for engine in engines:
        kill_servers()
        require_exclusive_gpu(f"before starting {engine}")
        proc, port = start_server(engine)
        # Our own server is expected to be present during the measurements.
        allow = ("llama-server",) if ENGINES[engine]["kind"] == "llama" \
            else ("./target/release/gb10-server",)
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
                                           args.trials, args.max_tokens, args.timeout,
                                           allow)
                # Accumulate rather than overwrite: naming the same engine twice
                # on the command line is how an arm gets interleaved with its
                # control (A,B,A,B) so monotonic within-session drift cancels
                # instead of loading onto whichever arm ran second. The reported
                # figure is the minimum over every repeat and trial.
                results.setdefault((engine, ctx), []).append((best, seen, reps))
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            # Reap it, or it lingers as `[gb10-server] <defunct>` -- which the
            # contention guard then sees by name on the next engine.
            try:
                proc.wait(timeout=30)
            except Exception:
                pass
            kill_servers()
            time.sleep(5)

    # ---- report -------------------------------------------------------------
    # `engines` may name the same arm twice (interleaved A,B,A,B); report each
    # distinct arm once, taking the minimum over every repeat and trial.
    names = []
    for e in engines:
        if e not in names:
            names.append(e)

    def best_of(v):
        return min(x[0] for x in v) if v else None

    def seen_of(v):
        for x in v:
            if x[1] is not None:
                return x[1]
        return None

    emit()
    emit("## Cold TTFT (s, minimum over every repeat and trial)")
    emit()
    emit("| context | " + " | ".join(names) + " |")
    emit("|---|" + "---|" * len(names))
    for ctx in contexts:
        cells = []
        for e in names:
            b = best_of(results.get((e, ctx)))
            cells.append(f"{b:.2f}" if b is not None else "n/a")
        emit(f"| {ctx} | " + " | ".join(cells) + " |")

    emit()
    emit("## Ratio vs the first arm (> 1.00 means the first arm is faster)")
    emit()
    # With llama.cpp present the ratio is gb10/llama; without it the first named
    # arm is the control, so the ratio is control/treatment, i.e. the speedup.
    base = "llama" if "llama" in engines else names[0]
    emit(f"| context | " + " | ".join(e for e in names if e != base) + " |")
    emit("|---|" + "---|" * (len(names) - 1))
    for ctx in contexts:
        b = best_of(results.get((base, ctx)))
        cells = []
        for e in names:
            if e == base:
                continue
            v = best_of(results.get((e, ctx)))
            cells.append(f"{b / v:.4f}" if (v and b) else "n/a")
        emit(f"| {ctx} | " + " | ".join(cells) + " |")

    emit()
    emit("## Every measurement (raw, in run order)")
    emit()
    emit("| # | context | arm | min cold TTFT s | prompt tokens | reps |")
    emit("|---|---|---|---|---|---|")
    n = 0
    for ctx in contexts:
        for e in engines:
            for (best, seen, reps) in results.get((e, ctx), []):
                n += 1
                emit(f"| {n} | {ctx} | {e} | {best:.2f} | {seen} | {reps} |")

    emit()
    emit("## Prompt tokens actually seen (proof both engines got the same prompt)")
    emit()
    emit("| context | " + " | ".join(names) + " |")
    emit("|---|" + "---|" * len(names))
    for ctx in contexts:
        cells = []
        for e in names:
            s = seen_of(results.get((e, ctx)))
            cells.append(str(s) if s is not None else "n/a")
        emit(f"| {ctx} | " + " | ".join(cells) + " |")

    emit()
    emit(f"- finished {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
