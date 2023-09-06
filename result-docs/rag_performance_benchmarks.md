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


## Llama 13B on A100

- A100 性能看起来和 3090 差不太多
### Llama 13B on A100 with num_context = 1

```bash
# default server MAX_BATCH_TOTAL_TOKENS
$ python bench_model.py bench-rag -j "results-tgi/mmlu*-1.jsonl" -t 8 -i 10 -m llama2-13b-a100
P1045478 2023-09-06 11:22:08,197 INFO [bench_model.py:45] Run RAG performance benchmark with config_file=model.yaml, model=llama2-13b-a100, jsonl_files=results-tgi/mmlu*-1.jsonl
P1045478 2023-09-06 11:22:18,384 INFO [bench_model.py:37] Throughput: 1.6 req/s, 617 prompt tokens/s (avg length 391), 157 completion tokens/s (avg_length 100)
P1045478 2023-09-06 11:22:28,469 INFO [bench_model.py:37] Throughput: 1.9 req/s, 589 prompt tokens/s (avg length 312), 188 completion tokens/s (avg_length 100)
P1045478 2023-09-06 11:22:39,015 INFO [bench_model.py:37] Throughput: 2.3 req/s, 716 prompt tokens/s (avg length 314), 227 completion tokens/s (avg_length 100)
P1045478 2023-09-06 11:22:49,217 INFO [bench_model.py:37] Throughput: 2.4 req/s, 730 prompt tokens/s (avg length 310), 235 completion tokens/s (avg_length 100)
P1045478 2023-09-06 11:22:59,688 INFO [bench_model.py:37] Throughput: 1.9 req/s, 691 prompt tokens/s (avg length 362), 190 completion tokens/s (avg_length 100)
P1045478 2023-09-06 11:23:10,105 INFO [bench_model.py:37] Throughput: 2.3 req/s, 653 prompt tokens/s (avg length 283), 230 completion tokens/s (avg_length 100)

# server MAX_BATCH_TOTAL_TOKENS=65536，大约还是 2.3 req/s 的吞吐量
python bench_model.py bench-rag -j "results-tgi/mmlu*-1.jsonl" -t 8 -i 10 -m llama2-13b-a100
P1046988 2023-09-06 11:25:13,517 INFO [bench_model.py:37] Throughput: 1.4 req/s, 542 prompt tokens/s (avg length 400), 135 completion tokens/s (avg_length 100)
P1046988 2023-09-06 11:25:23,769 INFO [bench_model.py:37] Throughput: 2.0 req/s, 585 prompt tokens/s (avg length 285), 204 completion tokens/s (avg_length 100)
P1046988 2023-09-06 11:25:33,871 INFO [bench_model.py:37] Throughput: 2.3 req/s, 716 prompt tokens/s (avg length 314), 227 completion tokens/s (avg_length 100)
P1046988 2023-09-06 11:25:44,129 INFO [bench_model.py:37] Throughput: 2.0 req/s, 621 prompt tokens/s (avg length 303), 204 completion tokens/s (avg_length 100)

# server MAX_BATCH_TOTAL_TOKENS=131072
python bench_model.py bench-rag -j "results-tgi/mmlu*-1.jsonl" -t 8 -i 10 -m llama2-13b-a100
P1048773 2023-09-06 11:28:40,216 INFO [bench_model.py:37] Throughput: 2.2 req/s, 714 prompt tokens/s (avg length 323), 220 completion tokens/s (avg_length 100)
P1048773 2023-09-06 11:28:50,839 INFO [bench_model.py:37] Throughput: 2.3 req/s, 710 prompt tokens/s (avg length 314), 225 completion tokens/s (avg_length 100)
```


### Llama 13B on A100 with num_context = 5

```bash
$ python bench_model.py bench-rag -j "results-tgi/mmlu*-5.jsonl" -t 8 -i 10 -m llama2-13b-a100
P1052250 2023-09-06 11:36:54,882 INFO [bench_model.py:37] Throughput: 1.7 req/s, 1409 prompt tokens/s (avg length 812), 173 completion tokens/s (avg_length 100)
P1052250 2023-09-06 11:37:04,940 INFO [bench_model.py:37] Throughput: 1.8 req/s, 1519 prompt tokens/s (avg length 849), 178 completion tokens/s (avg_length 100)

```