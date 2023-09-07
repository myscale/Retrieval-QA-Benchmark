# NOTE

You can run command below:

```shell
python bench_model.py bench-rag -x 0,1,2,3,5,10 -t 1,2,4,8,16,32,64,100 -i 30 -o 30,100,200 -m llama-2-13B-ensemble-v5-a100
```

Finally, the benchmark results looks like:

```json
{
    "llama2-13b-3090x2": {
        "1": {
            "100": {
                "64": {
                    "QA_Throughput": 4.707568657196474,
                    "QA_Latency": 61.86542979669571,
                    "Prompt_tokens_per_second": 1290.876524195817,
                    "Avg_prompt_length": 274.213,
                    "Completion_tokens_per_second": 430.49773856330324,
                    "Avg_completion_length": 91.448,
                    "Completion_latency": 676.5093801580757
                }
            }
        },
        "1": {
            "30": {
                "64": {
                    "QA_Throughput": 7.6156059335294835,
                    "QA_Latency": 39.74641561603546,
                    "Prompt_tokens_per_second": 2088.2981498509203,
                    "Avg_prompt_tokens": 274.213,
                    "Completion_tokens_per_second": 228.22447861601154,
                    "Avg_completion_tokens": 29.968,
                    "Completion_token_latency": 1326.295235452331
                }
            }
        }
    }
}
```

you can write your plot code to visualize the result.