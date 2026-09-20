#!/usr/bin/env bash
# Build llama.cpp with CUDA for GB10 (sm_121) as the M8 comparison baseline.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
fi
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 20 --target llama-cli llama-bench llama-server
echo "BUILD_OK"
