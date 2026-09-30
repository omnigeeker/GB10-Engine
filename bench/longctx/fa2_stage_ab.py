#!/usr/bin/env python3
"""Same-binary, same-session A/B of the FA2 prefill attention kernel.

Why a PTX swap and not an env var
---------------------------------
`crates/gb10-cuda/src/lib.rs:36` bakes the PTX *path* in at compile time
(`env!("GB10_KERNEL_PTX")`), but the file itself is read by
`ctx.load_module(Ptx::from_file(path))` when the process starts. So two kernels
can be compared by running the SAME binary twice with a different
`elementwise.ptx` in place. That is a stronger control than an env-var switch:
the Rust side, the launch configuration and every other kernel are literally
identical, and no extra kernel parameter or branch is introduced into the
kernel under test (which would perturb register allocation -- the thing that
decides occupancy here).

This is the mechanism the project already requires: only same-session pairs are
comparable (23% cross-session drift has been observed on this box), so all arms
are measured in one session, interleaved A,B,C,A,B,C to cancel monotonic drift,
and the minimum of the passes is reported.

Every run is bracketed by a GPU-ownership check: if any gb10 or llama process
appears, the run aborts rather than recording a contended number. A number
produced under contention looks like evidence and is not.
"""
import argparse
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODEL = "models/Qwen3.8-27B-NVFP4"
VERIFY = "./target/release/gb10-verify"

BUSY = ("gb10-verify", "gb10-server", "llama-server", "test-backend-ops")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def out_dir():
    """The single gb10-cuda OUT_DIR holding elementwise.ptx."""
    hits = glob.glob(os.path.join(ROOT, "target/release/build/gb10-cuda-*/out/elementwise.ptx"))
    if len(hits) != 1:
        sys.exit(f"expected exactly one elementwise.ptx, found {hits}")
    return os.path.dirname(hits[0])


def gpu_busy():
    """Return the list of foreign GPU processes, using `ps`, never `pgrep -f`
    (which self-matches its own command line and has produced phantom readings
    here).

    Zombies are excluded: a killed server whose parent has not reaped it shows
    as `[gb10-server] <defunct>`, which matches by name but holds no GPU
    resources. That false positive aborted the first 128K hoist A/B run.
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
        stat, _, args = rest.strip().partition(" ")
        if not pid.isdigit() or int(pid) == me:
            continue
        if stat.startswith("Z") or "<defunct>" in args or "grep" in args:
            continue
        if any(b in args for b in BUSY):
            hits.append(f"{pid} [{stat}] {args}")
    return hits


def swap_ptx(path):
    """Atomically install `path` as the OUT_DIR elementwise.ptx."""
    dst = os.path.join(out_dir(), "elementwise.ptx")
    tmp = dst + ".tmp"
    shutil.copyfile(path, tmp)
    os.replace(tmp, dst)
    return hashlib.sha256(open(dst, "rb").read()).hexdigest()[:12]


def run_once(ctx, extra_env=None):
    env = dict(os.environ)
    env.update({"GB10_FA2": "1", "GB10_PREFILL_NSEQ": "1", "GB10_ATTN_EVENTS": "1"})
    if extra_env:
        env.update(extra_env)
    cmd = [VERIFY, "prefill-shape", "--model", MODEL,
           "--limit", str(ctx), "--max-seq", str(ctx)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    dt = time.time() - t0
    if r.returncode != 0:
        return None
    txt = r.stdout
    m = re.search(r"attn kernel (\d+)ms", txt)
    t = re.search(r"total ([\d.]+)s", txt)
    return {"attn_ms": int(m.group(1)) if m else None,
            "total_s": float(t.group(1)) if t else None,
            "wall_s": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True,
                    help="comma-separated name=ptxpath, optionally with a "
                         "per-arm env suffix: name=ptxpath@KEY=VAL;KEY2=VAL2. "
                         "The env travels with the arm so a knob that must "
                         "match the kernel (e.g. the host's dynamic smem "
                         "request) cannot be desynchronised from it.")
    ap.add_argument("--contexts", default="8192,32768")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms = []
    for spec in args.arms.split(","):
        name, _, rest = spec.partition("=")
        path, _, envspec = rest.partition("@")
        env = {}
        for kv in envspec.split(";"):
            kv = kv.strip()
            if kv:
                k, _, v = kv.partition("=")
                env[k.strip()] = v.strip()
        if not path or not os.path.isfile(path):
            sys.exit(f"bad arm {spec!r}")
        arms.append((name.strip(), path, env))
    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]

    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("# FA2 staging-loop A/B -- same binary, PTX swap, one session")
    emit()
    emit(f"- started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    emit(f"- passes {args.passes} per arm per context, interleaved, minimum reported")
    emit(f"- instrument: `gb10-verify prefill-shape`, GB10_FA2=1 GB10_PREFILL_NSEQ=1 "
         f"GB10_ATTN_EVENTS=1, chunk 8192 (the server's PREFILL_CHUNK)")
    emit(f"- contexts {contexts}")
    emit()
    emit("| arm | ptx sha256[:12] | file | extra env |")
    emit("|---|---|---|---|")
    shas = {}
    for name, path, env in arms:
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
        shas[name] = h
        es = " ".join(f"`{k}={v}`" for k, v in sorted(env.items())) or "-"
        emit(f"| {name} | `{h}` | {os.path.basename(path)} | {es} |")
    emit()

    results = {}
    for ctx in contexts:
        for p in range(args.passes):
            for name, path, env in arms:
                busy = gpu_busy()
                if busy:
                    sys.exit(f"GPU BUSY before {name} ctx={ctx} pass={p}: {busy}")
                got = swap_ptx(path)
                assert got == shas[name], f"ptx sha mismatch {got} != {shas[name]}"
                log(f"ctx={ctx} pass={p} arm={name} ptx={got} ...")
                r = run_once(ctx, env)
                if r is None or r["attn_ms"] is None:
                    log(f"  FAILED: {r}")
                    continue
                log(f"  attn kernel {r['attn_ms']}ms  total {r['total_s']}s")
                results.setdefault((name, ctx), []).append(r)
                # A contended run is worse than no run: check nothing appeared.
                busy = gpu_busy()
                if busy:
                    sys.exit(f"GPU CONTENDED during {name} ctx={ctx} pass={p}: {busy}")

    emit("## attn kernel (ms, minimum of passes)")
    emit()
    emit("| context | " + " | ".join(n for n, _, _ in arms) + " |")
    emit("|---|" + "---|" * len(arms))
    for ctx in contexts:
        cells = []
        for n, _, _ in arms:
            v = results.get((n, ctx))
            cells.append(f"{min(x['attn_ms'] for x in v)}" if v else "n/a")
        emit(f"| {ctx} | " + " | ".join(cells) + " |")
    emit()
    emit("## attn kernel speedup vs the first arm")
    emit()
    base = arms[0][0]
    emit("| context | " + " | ".join(n for n, _, _ in arms if n != base) + " |")
    emit("|---|" + "---|" * (len(arms) - 1))
    for ctx in contexts:
        b = results.get((base, ctx))
        bm = min(x["attn_ms"] for x in b) if b else None
        cells = []
        for n, _, _ in arms:
            if n == base:
                continue
            v = results.get((n, ctx))
            cells.append(f"{bm / min(x['attn_ms'] for x in v):.4f}"
                         if (v and bm) else "n/a")
        emit(f"| {ctx} | " + " | ".join(cells) + " |")
    emit()
    emit("## all passes (raw)")
    emit()
    emit("| context | arm | attn kernel ms | total s |")
    emit("|---|---|---|---|")
    for ctx in contexts:
        for n, _, _ in arms:
            for x in results.get((n, ctx), []):
                emit(f"| {ctx} | {n} | {x['attn_ms']} | {x['total_s']} |")
    emit()

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
