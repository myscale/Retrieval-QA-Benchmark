import json
import time
import sys
import os
import types
from typing import List, Union
import logging.config
from glob import glob
from clize import run
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import multiprocessing
from threading import Timer
from retrieval_qa_benchmark.schema import QARecord, QAPrediction
from retrieval_qa_benchmark.utils.profiler import PROFILER
from functools import partial
from transformers import AutoTokenizer

from retrieval_qa_benchmark.models import *
from retrieval_qa_benchmark.schema import *
from retrieval_qa_benchmark.utils.factory import ModelFactory
from retrieval_qa_benchmark.utils.config import load
logging.config.fileConfig("logging.conf")

def load_jsonl(fn):
    with open(fn) as f:
        return [json.loads(s) for s in f.readlines()[2:]]

def report_stats(records: List[QAPrediction], profile_name:str,  pre_time: float, need_write: bool, model_name: str, context: int, max_new_token:int, thread:int, results_detail_file_name:str, results_overview_file_name:str):
    # 本次统计的时间间隔
    duration = time.time() - pre_time
    records_size = len([record for record in records if record["time"]>=pre_time])

    # 输入的总 token 数量
    prompt_tokens = [record["prompt_tokens"] for record in records if record["time"]>=pre_time]
    # 输出的总 token 数量
    completion_tokens = [record["completion_tokens"] for record in records if record["time"]>=pre_time]
    # 所有 QA boot 总耗时 / s
    total_boot_time = sum(record["profile_time"]['boot_time'] for record in records if record["time"]>=pre_time) / 1000
    # 所有 QA profile 总耗时 / s
    total_profile_time = sum(record["profile_time"][profile_name] for record in records if record["time"]>=pre_time) / 1000
    # 所有 QA profile 总平均耗时 / s
    total_profile_avg_time = sum(record["profile_avg"][profile_name] for record in records if record["time"]>=pre_time) / 1000

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
    qa_latency = (total_profile_time + total_boot_time) / records_size
    # QA 输出 token 时延 / ms
    completion_latency = (total_profile_time / (sum(completion_tokens) - records_size)) * 1000


    logging.info("%s Throughput: %.1f req/s, Latency: %.1f s/req, %d prompt tokens/s (avg length %d), %d completion tokens/s (avg_length %d), completion latency %.3f ms",
                 "Totally" if need_write else "", qa_throughput, qa_latency, prompt_throughput, avg_prompt_tokens, completion_throughput, avg_completion_tokens, completion_latency)
    
    stats = {
        "QA_Throughput": qa_throughput,
        "QA_Latency": qa_latency,

        "Prompt_tokens_per_second": prompt_throughput,
        "Avg_prompt_tokens": avg_prompt_tokens,
        
        "Completion_tokens_per_second": completion_throughput,
        "Avg_completion_tokens": avg_completion_tokens,
        
        "Completion_token_latency": completion_latency,

        "details": records
    }

    overview = {
        "model_name": model_name,
        "context": context,
        "max_new_token": max_new_token,
        "thread": thread,
        "QA_QPS": round(qa_throughput,2),
        "QA_Latency": round(qa_latency,2),
        "Prompt_QPS": round(prompt_throughput,2),
        "Prompt_AVG": round(avg_prompt_tokens,2),
        "Completion_QPS": round(completion_throughput,2),
        "Completion_AVG": round(avg_completion_tokens,2),
        "Completion_Latency": round(completion_latency,2)
    }

    if need_write:
        # 读取已有的JSON数据
        data = {}
        overviews = []
        if os.path.exists(results_detail_file_name):
            with open(results_detail_file_name, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # 文件为空或包含无效的 JSON，所以保持 data 为 {}
                    data = {}
        if os.path.exists(results_overview_file_name):
            with open(results_overview_file_name, 'r') as f:
                try:
                    overviews = json.load(f)
                except json.JSONDecodeError:
                    # 文件为空或包含无效的 JSON，所以初始化为空数组
                    overviews = []

        # 更新JSON数据的层级结构
        # print(data)
        if model_name not in list(data.keys()):
            data[model_name] = {}
        if str(context) not in list(data[model_name].keys()):
            data[model_name][str(context)] = {}
        if str(max_new_token) not in list(data[model_name][str(context)].keys()):
            data[model_name][str(context)][str(max_new_token)] = {}
        data[model_name][str(context)][str(max_new_token)][str(thread)] = stats
        overviews.append(overview)

        # 写回JSON文件
        with open(results_detail_file_name, 'w') as f:
            json.dump(data, f, indent=2)
        with open(results_overview_file_name, 'w') as f:
            f.write('[\n')
            for index, item in enumerate(overviews):
                f.write(json.dumps(item, indent=None))
                # 只有当它不是最后一个项目时，才添加逗号
                if index != len(overviews) - 1:
                    f.write(',')
                f.write('\n')
            f.write(']\n')


# 定义调用 model 的函数
def single(in_: Union[QARecord, int], model: TGI_LLM, profile_name: str):
    r, prompt_tokens = in_
    # PROFILER.clear()
    q = model.convert_record(r)
    pred_str = ""
    cnt = 0
    t0 = time.time()
    stream = model.client.text_generation(q, stream=True, details=True, **model.run_args)
    for i, token in enumerate(stream):
        if i == 0:
            t_boot = time.time()
        if not token.token.special:
            pred_str += token.token.text
        cnt += 1
    t_gen = (time.time() - t_boot) * 1000
    t_boot = (t_boot - t0) * 1000
    pred = BaseLLMOutput(
            generated=pred_str,
            completion_tokens=cnt,
            prompt_tokens=prompt_tokens,
        ).model_dump()
    accumulator = {'boot_time': t_boot, profile_name: t_gen}
    
    pred.update({'profile_time': accumulator})
    pred.update({'profile_count': {k: 1 for k in accumulator}})
    pred.update({'profile_avg': accumulator})
    return pred

def bench_rag(*, 
              max_records:'n'=1000,
              num_threads:'p'="1,4,8",
              max_new_tokens:'o'="30,100",
              contexts:'x'="0,1,5",
              config_file:'c'="model.yaml",
              models:'m'="default",
              report_interval:'i'=30,
              results_detail_file_name:'r1'="bench_results_detail.json",
              results_overview_file_name:'r2'="bench_results_overview.json",
              time_out:'t'=120,  # 单个试验运行最长耗时 / s
            ):
    logging.info(f"Run RAG performance benchmark with config_file={config_file}, models={models}, contexts={contexts}")
    
    # 加载配置文件
    with open(config_file) as f:
        config = load(f)
    
    # 清空历史结果
    with open(results_detail_file_name, 'w') as f:
        pass
    with open(results_overview_file_name, 'w') as f:
        pass

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
                        model: TGI_LLM = ModelFactory.from_config(model_config).build()
                        
                        # 过滤相关的 records
                        jsonl_files =f"results-tgi/mmlu*-{context}.jsonl"
                        records = []
                        for f in glob(jsonl_files):
                            records.extend(load_jsonl(f))
                        prompt_tokens = [r['prompt_tokens'] for r in records]
                        records = [QARecord(**{k: v for k, v in r.items() if k in QARecord.model_fields.keys()}) for r in records]
                        records = records[:max_records]

                        
                        

                        def process_records(pre_time: float, time_out: float):
                            res = []
                            # 创建一个共享变量，作为终止标志
                            terminate_flag = multiprocessing.Value('i', 0)
                            with multiprocessing.Pool(thread) as p:  # 注意这里使用了multiprocessing.Pool
                                try:
                                    child_pids = [worker.pid for worker in p._pool]
                                    logging.info(f"Current PID: {os.getpid()}, Child process PIDs:{child_pids}")
                                except Exception as e:
                                    pass
                                # 进程池终止的逻辑
                                def terminate_pool():
                                    with terminate_flag.get_lock():
                                        terminate_flag.value = 1
                                # 设定进程池终止计时器
                                timer = Timer(time_out, terminate_pool)
                                timer.start()  # 开始计时
                                partial_single = partial(single, model=model, profile_name=profile_name)
                                try:
                                    for pred in tqdm(map(partial_single, zip(records, prompt_tokens)), total=len(records)):
                                        current_time = time.time()
                                        pred["time"] = current_time
                                        res.append(pred)
                                        # 判断是否超时
                                        if terminate_flag.value == 1:
                                            p.terminate()
                                            timer.cancel()
                                            sys.stdout.write("\r\033[K")
                                            logging.warning(f"[Processing Pool Terminating] early due to {time_out} sec timeout.")
                                            break
                                        # 间隔 report_interval 输出报告信息
                                        if report_interval > 0 and current_time - pre_time > report_interval:
                                            sys.stdout.write("\r\033[K")
                                            report_stats(res, profile_name, pre_time, False, model_name, context, max_new_token, thread, results_detail_file_name, results_overview_file_name)
                                            pre_time = current_time
                                        
                                finally:
                                    timer.cancel()
                            return res

                        bench_start_time = time.time()
                        result_records = process_records(pre_time=bench_start_time, time_out=time_out)
                        
                        # 写入结果到文件
                        report_stats(result_records, profile_name, bench_start_time, True, model_name, context, max_new_token, thread, results_detail_file_name, results_overview_file_name)
                    except Exception as e:
                        logging.error(f"Exception occur: {e}")
                        # raise e


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
