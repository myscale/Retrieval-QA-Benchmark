# RAG Performance Benchmarks

## Llama 13B on 3090x2

- 5 个 context 只比 1 个 context 慢了 30% 左右。

```bash
# 1 context
$ python bench_model.py bench-rag -j "results-tgi/mmlu*-1.jsonl" -t 8 -i 10

P669678 2023-09-05 20:09:03,974 INFO [bench_model.py:45] Run RAG performance benchmark with config_file=model.yaml, jsonl_files=results-tgi/mmlu*-1.jsonl
P669678 2023-09-05 20:09:15,100 INFO [bench_model.py:37] Throughput: 1.7 req/s, 634 prompt tokens/s (avg length 370), 153 completion tokens/s (avg_length 89)
P669678 2023-09-05 20:09:26,338 INFO [bench_model.py:37] Throughput: 2.2 req/s, 599 prompt tokens/s (avg length 269), 219 completion tokens/s (avg_length 98)
P669678 2023-09-05 20:09:37,891 INFO [bench_model.py:37] Throughput: 2.2 req/s, 730 prompt tokens/s (avg length 337), 212 completion tokens/s (avg_length 98)

# 5 contexts
$ python bench_model.py bench-rag -j "results-tgi/mmlu*-5.jsonl" -t 8 -i 10

P668872 2023-09-05 20:07:54,586 INFO [bench_model.py:45] Run RAG performance benchmark with config_file=model.yaml, jsonl_files=results-tgi/mmlu*-5.jsonl
P668872 2023-09-05 20:08:04,967 INFO [bench_model.py:37] Throughput: 1.0 req/s, 858 prompt tokens/s (avg length 886), 87 completion tokens/s (avg_length 90)
P668872 2023-09-05 20:08:15,620 INFO [bench_model.py:37] Throughput: 1.6 req/s, 1358 prompt tokens/s (avg length 851), 156 completion tokens/s (avg_length 97)
P668872 2023-09-05 20:08:25,894 INFO [bench_model.py:37] Throughput: 1.7 req/s, 1571 prompt tokens/s (avg length 949), 161 completion tokens/s (avg_length 97)
P668872 2023-09-05 20:08:36,524 INFO [bench_model.py:37] Throughput: 1.5 req/s, 1275 prompt tokens/s (avg length 847), 146 completion tokens/s (avg_length 97)
P668872 2023-09-05 20:08:46,782 INFO [bench_model.py:37] Throughput: 1.8 req/s, 1504 prompt tokens/s (avg length 857), 163 completion tokens/s (avg_length 93)
```


## Llama 13B on Huggingface A100