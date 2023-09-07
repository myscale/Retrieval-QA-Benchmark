import json
import time
import sys
import os
import types
from typing import List
import logging.config
from glob import glob
from clize import run
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from retrieval_qa_benchmark.schema import QARecord, QAPrediction
from retrieval_qa_benchmark.utils.profiler import PROFILER

from retrieval_qa_benchmark.models import *
from retrieval_qa_benchmark.schema import *
from retrieval_qa_benchmark.utils.factory import ModelFactory
from retrieval_qa_benchmark.utils.config import load
logging.config.fileConfig("logging.conf")

def load_jsonl(fn):
    with open(fn) as f:
        return [json.loads(s) for s in f.readlines()[2:]]

def report_stats(records: List[QAPrediction], profile_name:str,  t_prev: float, pre_size: int, need_write: bool, model_name: str, context: int, max_new_token:int, thread:int, results_file_name:str):
    # 本次统计的时间间隔
    duration = time.time() - t_prev
    records_size = len(records) - pre_size

    # 输入的总 token 数量
    prompt_tokens = [r["prompt_tokens"] for r in records[pre_size:]]
    # 输出的总 token 数量
    completion_tokens = [r["completion_tokens"] for r in records[pre_size:]]
    # 所有 QA profile 总耗时 / s
    total_profile_time = sum(record["profile_time"][profile_name] for record in records[pre_size:]) / 1000
    # 所有 QA profile 总平均耗时 / s
    total_profile_avg_time = sum(record["profile_avg"][profile_name] for record in records[pre_size:]) / 1000

    print(f"total_profile_time:{total_profile_time}, total_profile_avg_time:{total_profile_avg_time}, records_size:{records_size}")
    # 单条 QA 输入的 token 平均数量
    avg_prompt_tokens = sum(prompt_tokens) / records_size
    # 单条 QA 输出的 token 平均数量
    avg_completion_tokens = sum(completion_tokens) / records_size

    # QA 吞吐量
    qa_throughput = records_size / duration
    # QA 输入 token 吞吐量
    prompt_throughput = sum(prompt_tokens) / duration
    # QA 输出 token 吞吐量
    completion_throughput = sum(completion_tokens) / duration

    # QA 时延 / s
    qa_latency = total_profile_time / records_size
    # QA 输出 token 时延 / ms
    completion_latency = (total_profile_time / sum(completion_tokens)) * 1000


    logging.info("Throughput: %.1f req/s, Latency: %.1f s/req, %d prompt tokens/s (avg length %d), %d completion tokens/s (avg_length %d), completion latency %.3f ms",
                 qa_throughput, qa_latency, prompt_throughput, avg_prompt_tokens, completion_throughput, avg_completion_tokens, completion_latency)
    
    stats = {
        "QA_Throughput": qa_throughput,
        "QA_Latency": qa_latency,

        "Prompt_tokens_per_second": prompt_throughput,
        "Avg_prompt_tokens": avg_prompt_tokens,
        
        "Completion_tokens_per_second": completion_throughput,
        "Avg_completion_tokens": avg_completion_tokens,
        
        "Completion_token_latency": completion_latency
    }

    if need_write:
        # 读取已有的JSON数据
        data = {}
        if os.path.exists(results_file_name):
            with open(results_file_name, 'r') as f:
                data = json.load(f)

        # 更新JSON数据的层级结构
        if model_name not in data:
            data[model_name] = {}
        if context not in data[model_name]:
            data[model_name][context] = {}
        if max_new_token not in data[model_name][context]:
            data[model_name][context][max_new_token] = {}
        data[model_name][context][max_new_token][thread] = stats

        # 写回JSON文件
        with open(results_file_name, 'w') as f:
            json.dump(data, f, indent=4)



def bench_rag(*, 
              max_records:'n'=1000,
              num_threads:'t'="1,4,8",
              max_new_tokens:'o'="30,100",
              contexts:'x'="0,1,5",
              config_file:'c'="model.yaml",
              models:'m'="default",
              report_interval:'i'=30,
              results_file_name:'r'="bench_results.json",
            ):
    logging.info("Run RAG performance benchmark with config_file=%s, models=%s, contexts=%s", config_file, models, contexts)
    
    # 加载配置文件
    with open(config_file) as f:
        config = load(f)

    # 预处理用户输入的参数
    num_threads_list = [int(t.strip()) for t in num_threads.split(',')]
    contexts_list = [int(x.strip()) for x in contexts.split(',')]
    max_new_tokens_list = [int(o.strip()) for o in max_new_tokens.split(',')]
    model_name_list = [str(o.strip()) for o in models.split(',')]

    for model_name in model_name_list:
        for context in contexts_list:
            for max_new_token in max_new_tokens_list:
                for thread in num_threads_list:
                    try:
                        logging.info(f"Sub benchmark: model_name:{model_name}, context:{context}, max_new_token:{max_new_token}, thread:{thread}")
                    
                        # 初始化 model 相关信息
                        model_config = config[model_name]
                        model_type = model_config["type"]
                        profile_name = f"model.{model_type}.profile"
                        model_config["run_args"]["max_new_tokens"] = max_new_token
                        model: BaseLLM = ModelFactory.from_config(model_config).build()
                        
                        # 过滤相关的 records
                        jsonl_files =f"results-tgi/mmlu*-{context}.jsonl"
                        records = []
                        for f in glob(jsonl_files):
                            records.extend(load_jsonl(f))
                        records = [QARecord(**{k: v for k, v in r.items() if k in QARecord.model_fields.keys()}) for r in records]
                        records = records[:max_records]

                        # 定义调用 model 的函数
                        def single(r:QARecord):
                            PROFILER.clear()
                            pred = model.generate(r).model_dump()
                            pred.update({'profile_time': PROFILER.accumulator})
                            pred.update({'profile_count': PROFILER.counter})
                            pred.update({'profile_avg': {profile_name: (PROFILER.accumulator[profile_name]/PROFILER.counter[profile_name])}})
                            return pred
                        
                        result_records = []
                        t0 = time.time()
                        t_last = t0
                        pre_size = 0  # result_records 的前一次长度
                        with ThreadPool(thread) as p:
                            for pred in tqdm(p.imap_unordered(single, records), total=len(records)):
                                t = time.time()
                                pred["time"] = t
                                result_records.append(pred)
                                if report_interval > 0 and t - t_last > report_interval:
                                    # clear current line
                                    sys.stdout.write("\r\033[K")
                                    report_stats(result_records, profile_name, t_last, pre_size, False, model_name, context, max_new_token, thread, results_file_name)
                                    t_last = t
                                    pre_size = len(result_records)
                        report_stats(result_records, profile_name, t0, 0, True, model_name, context, max_new_token, thread, results_file_name)
                    except Exception as e:
                        logging.error(f"Exception occur: {e}")


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
