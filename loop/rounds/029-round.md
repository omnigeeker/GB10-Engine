# Round 29 — 20260920T071735Z

## Gates

| gate | result |
|---|---|
| build | fail |
| test | skip |
| correctness (layers) | skip |
| correctness (64-layer) | skip |
| benchmark | skip |

**status: FAIL**

## Log

```
    |             help: remove this `mut`
    |
    = note: `#[warn(unused_mut)]` (part of `#[warn(unused)]`) on by default

warning: fields `max_abs` and `worst` are never read
  --> crates/gb10-verify/src/main.rs:87:5
   |
86 | struct Diff {
   |        ---- fields in this struct
87 |     max_abs: f32,
   |     ^^^^^^^
...
91 |     worst: usize,
   |     ^^^^^
   |
   = note: `#[warn(dead_code)]` (part of `#[warn(unused)]`) on by default

   Compiling gb10-bench v0.1.0 (/home/wayne/dsh/QWen3.8-27B-GB10/crates/gb10-bench)
   Compiling gb10-server v0.1.0 (/home/wayne/dsh/QWen3.8-27B-GB10/crates/gb10-server)
warning: `gb10-verify` (bin "gb10-verify") generated 21 warnings (run `cargo fix --bin "gb10-verify" -p gb10-verify` to apply 1 suggestion)
error[E0061]: this function takes 4 arguments but 3 arguments were supplied
   --> crates/gb10-server/src/main.rs:194:21
    |
194 |         let state = ModelState::new(&dev, &model, MAX_SEQ)?;
    |                     ^^^^^^^^^^^^^^^----------------------- argument #4 of type `usize` is missing
    |
note: associated function defined here
   --> crates/gb10-model/src/model.rs:104:12
    |
104 |     pub fn new(dev: &Device, model: &Model, max_seq: usize, n_seq: usize) -> Result<Self> {
    |            ^^^
help: provide the argument
    |
194 |         let state = ModelState::new(&dev, &model, MAX_SEQ, /* usize */)?;
    |                                                          +++++++++++++

For more information about this error, try `rustc --explain E0061`.
error: could not compile `gb10-server` (bin "gb10-server") due to 1 previous error
warning: build failed, waiting for other jobs to finish...
[round 29] build FAILED (see /home/wayne/dsh/QWen3.8-27B-GB10/loop/logs/round-029-20260920T071735Z.log)
```
