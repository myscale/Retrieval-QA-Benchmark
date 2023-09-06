#!/bin/bash

THREADS_LIST=(4 8 16 32 64 128)
ITERATIONS=-1
MODEL="llama-2-13B-ensemble-v5"
CONTEXTS=(1 2 3 5 10)

for THREADS in "${THREADS_LIST[@]}"; do
    for context in "${CONTEXTS[@]}"; do
        echo "thread: ${THREADS}, context: ${context}"
        python3 bench_model.py bench-rag -j "results-tgi/mmlu*-${context}.jsonl" -t "${THREADS}" -i "${ITERATIONS}" -m "${MODEL}"
    done
done
