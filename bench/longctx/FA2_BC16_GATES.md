# BC=16 kernel (`e00754e5f1c1`) — gates and memory safety

Kernel under test: `target/release/build/gb10-cuda-*/out/elementwise.ptx`,
sha256 `e00754e5f1c163d9d098a2ccf130e2de4d341befee15135560248ff1a1e24dd0`
(`FA2_BC = 16` in `kernels/elementwise.cu:703`; host `GB10_FA2_BC` default 16 in
`crates/gb10-cuda/src/ops.rs:2058`, which must match or occupancy silently drops).

## Gates — baseline (BC=32 / pre-BC=16) vs BC=16

| gate | baseline | **BC=16** | delta | verdict |
|---|---|---|---|---|
| `generate` (ids) | exact 16/16 | **exact 16/16** | 0 | pass |
| `perplexity --ctx 512 --chunks 60` | mean nll 1.875067 | **1.875132** | **+6.5e-5** (+0.0035%) | pass, disclosed |
| `perplexity --ctx 4096 --chunks 60` | mean nll 1.877331 | **1.877423** | **+9.2e-5** (+0.0049%) | pass, disclosed |
| `attn-tile` | 12 MISMATCH / 13 shapes | **same 12 / same 13, 0 shapes >1.5x worse** | identical profile | rc=1 **by design** |

`generate` exact ids, reproduced this session from `/tmp/gates_bc16/generate.txt`:

```
[1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]
oracle agreement: 16/16 (100.0%)   exact match
```

The two perplexity deltas are **not** bit-identical, and that is the accepted, characterised cost
of the BC=16 win — see `HANDOFF.md` "THE GATE STANDARD IS RESTATED". The online softmax runs over
key tiles, so halving the tile width changes the running-max/rescale order and rescales the fp16
`P*V` accumulator **twice as often**. That explanation predicts both gates move slightly *worse*,
which is what is recorded. **An fp32 `P*V` accumulator would buy the 5e-5 back and cost the
register budget holding 4 CTAs/SM, i.e. the entire 1.33–1.47x. Do not "fix" it.**

`attn-tile` is a **differential** test (fp16 accumulator by design) and must not veto a change that
`generate` and `perplexity` pass. It returns **rc=1 by design**.

## Memory safety — `compute-sanitizer` on the BC=16 kernel

Run **this session** against the installed BC=16 PTX, on `attn-tile` (the indexing-heavy path):

```
compute-sanitizer --tool memcheck   --report-api-errors no --error-exitcode 9 \
    ./target/release/gb10-verify attn-tile --model models/Qwen3.8-27B-NVFP4
  -> ERROR SUMMARY: 0 errors                    (rc=1: attn-tile's by-design MISMATCH)

compute-sanitizer --tool racecheck  --report-api-errors no --error-exitcode 9 \
    ./target/release/gb10-verify attn-tile --model models/Qwen3.8-27B-NVFP4
  -> RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

**Both clean.** Raw output committed at `bench/longctx/gates_bc16/sanitizer_memcheck.txt`
and `bench/longctx/gates_bc16/sanitizer_racecheck.txt` (also in `/tmp/gates_bc16/`).

`--report-api-errors no` is **required**, not a weakening: `gb10_cuda::Device::new` looks up 9
kernel names that live in a different PTX module, so every Device creation emits 9
`CUDA_ERROR_NOT_FOUND` **API** errors — pre-existing, independent of the kernel, and counted in
`ERROR SUMMARY` by default (~114 across a run). Without the flag the gate is a false failure in an
abort-by-default harness.

**Why this check matters here:** the correctness gates do **not** certify memory safety — an
out-of-bounds store once passed `generate` 16/16 *and* perplexity. `compute-sanitizer` is the real
check for anything touching indexing, and BC=16 changed the tile indexing.

## Reproduced this session vs carried forward

| item | this session | source |
|---|---|---|
| `generate` exact 16/16 | re-read and confirmed | `/tmp/gates_bc16/generate.txt` |
| `ppl512` mean nll 1.875132 | re-read and confirmed | `/tmp/gates_bc16/ppl512.txt` |
| `ppl4096` mean nll 1.877423 | **carried forward** — the local file is truncated at window 30; the value is committed at `bench/longctx/comparison.md:15146` | `comparison.md:15146` |
| `attn-tile` 12 MISMATCH / 13 shapes | **re-executed** under both sanitizer tools; 12 MISMATCH rows counted | `/tmp/gates_bc16/sanitizer_*.txt` |
| sanitizer memcheck + racecheck | **new this session, both clean** | `bench/longctx/gates_bc16/sanitizer_*.txt` |

Raw gate outputs for this kernel are committed alongside this file in
`bench/longctx/gates_bc16/` (`generate.txt`, `ppl512.txt`, `ptx.sha256`, `sanitizer_*.txt`).
`ppl4096.txt` is **not** included because the local capture is truncated at window 30; the
committed value of record is `bench/longctx/comparison.md:15146`.
