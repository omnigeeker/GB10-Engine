# Round 3 — 20260918T113128Z

## Gates

| gate | result |
|---|---|
| build | pass |
| test | pass |
| correctness | missing |
| benchmark | pass |

**status: PASS**

## Log

```
     Running unittests src/main.rs (target/release/deps/gb10_server-f4cd4cdf35b7008a)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests gb10_core

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests gb10_cuda

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests gb10_model

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

[round 3] tests OK
[round 3] gb10-verify not built yet — gate pending
[round 3] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 39.8s

per-token weight traffic : 17.555 GB
time per token           : 73.35 ms
achieved bandwidth       : 239.3 GB/s
projected decode         : 13.63 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 105.0% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260918T113128Z.json
[round 3] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260918T113128Z.json
```
