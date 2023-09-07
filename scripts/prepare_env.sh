#!/bin/bash
apt-get update
apt-get install --yes vim unzip wget tar

wget https://mqdb-release-1253802058.cos.ap-beijing.myqcloud.com/RQA/RQA.tar.gz
tar -zxvf RQA.tar.gz
cd RQA
wget https://mqdb-release-1253802058.cos.ap-beijing.myqcloud.com/RQA/results-tgi.zip
unzip results-tgi.zip
python3 -m pip install -e .

# you can run scripts below:
# python3 bench_model.py bench-rag -j "results-tgi/mmlu*-1.jsonl" -t 8 -i 10 -m llama-2-13B-ensemble-v5-a100