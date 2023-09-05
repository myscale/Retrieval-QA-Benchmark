import json
import time
import sys
import types
import logging.config
from glob import glob
from clize import run
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from retrieval_qa_benchmark.schema import QARecord
from retrieval_qa_benchmark.utils.profiler import PROFILER

from retrieval_qa_benchmark.models import *
from retrieval_qa_benchmark.schema import *
from retrieval_qa_benchmark.utils.factory import ModelFactory
from retrieval_qa_benchmark.utils.config import load
logging.config.fileConfig("logging.conf")

def load_jsonl(fn):
    with open(fn) as f:
        return [json.loads(s) for s in f.readlines()[2:]]

def report_stats(records, t_prev):
    duration = time.time() - t_prev
    prompt_tokens = []
    completion_tokens = []
    for r in records[::-1]:
        if r["time"] < t_prev: break
        prompt_tokens.append(r["prompt_tokens"])
        completion_tokens.append(r["completion_tokens"])
    num_req = len(prompt_tokens)
    req_throughput = num_req / duration
    avg_prompt_len = sum(prompt_tokens) / num_req
    avg_completion_len = sum(completion_tokens) / num_req
    prompt_throughput = sum(prompt_tokens) / duration
    completion_throughput = sum(completion_tokens) / duration
    logging.info("Throughput: %.1f req/s, %d prompt tokens/s (avg length %d), "
                 "%d completion tokens/s (avg_length %d)",
                 req_throughput, prompt_throughput, avg_prompt_len,
                 completion_throughput, avg_completion_len)

def bench_rag(*, max_records:'n'=1000, num_threads:'t'=4,
              config_file:'c'="model.yaml", max_new_tokens:'m'=100,
              jsonl_files:'j'="results-tgi/mmlu*.jsonl", report_interval:'i'=30):
    logging.info("Run RAG performance benchmark with config_file=%s, jsonl_files=%s",
                 config_file, jsonl_files)

    with open(config_file) as f:
        config = load(f)
    config["run_args"]["max_new_tokens"] = max_new_tokens
    model: BaseLLM = ModelFactory.from_config(config).build()

    records = []
    for f in glob(jsonl_files):
        records.extend(load_jsonl(f))

    records = [QARecord(**{k: v for k, v in r.items() if k in QARecord.model_fields.keys()})
               for r in records]
    records = records[:max_records]

    def single(r):
        PROFILER.clear()
        pred = model.generate(r).model_dump()
        pred.update({'profile_avg': PROFILER.accumulator})
        return pred

    result_records = []
    t0 = time.time()
    t_last = t0
    with ThreadPool(num_threads) as p:
        for pred in tqdm(p.imap_unordered(single, records), total=len(records)):
            t = time.time()
            pred["time"] = t
            result_records.append(pred)
            if report_interval > 0 and t - t_last > report_interval:
                # clear current line
                sys.stdout.write("\r\033[K")
                report_stats(result_records, t_last)
                t_last = t

    report_stats(result_records, t0)

if __name__ == '__main__':

    functions = {k: v for k, v in globals().items() if callable(v) and type(v) == types.FunctionType}
    try:
        run(functions, description="RAG performance benchmark toolkit")
    except Exception as e:
        import pdb
        import traceback
        traceback.print_exc()
        pdb.post_mortem()
        raise e
