import json
import time
from glob import glob
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from retrieval_qa_benchmark.schema import QARecord
from retrieval_qa_benchmark.utils.profiler import PROFILER

from retrieval_qa_benchmark.models import *
from retrieval_qa_benchmark.schema import *
from retrieval_qa_benchmark.utils.factory import ModelFactory
from retrieval_qa_benchmark.utils.config import load
config = load(open('model.yaml'))
model: BaseLLM = ModelFactory.from_config(config).build()

def load_jsonl(fn):
    with open(fn) as f:
        return [json.loads(s) for s in f.readlines()[2:]]

records = []
for f in glob('results-tgi/mmlu*.jsonl'):
    records.extend(load_jsonl(f))

records = [QARecord(**{k: v for k, v in r.items() if k in QARecord.model_fields.keys()}) for r in records]

new_records = []

def single(r):
    PROFILER.clear()
    pred = model.generate(r).model_dump()
    pred.update({'profile_avg': PROFILER.accumulator})
    return pred

t0 = time.time()
# 最大 batch_size * 4, 这里使用 batch_size = 4 为例
with ThreadPool(16) as p:
    for pred in tqdm(p.imap_unordered(single, records), total=len(records)):
        new_records.append(pred)
    t_delta = time.time() - t0

# 也可以选择把 `new_records` 存下来

total_tokens = [r['completion_tokens'] + r['prompt_tokens'] for r in new_records]

throughput_to_queries = 1 / t_delta
throughput_to_tokens = len(total_tokens) / t_delta # 如此也可以计算 生成吞吐 / 输入吞吐
