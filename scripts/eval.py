import os
from os import environ, path

# environ['HF_HOME'] = '/Volumes/DATA/hf_home'
environ["HF_HOME"] = "/mnt/workspaces/hf_home"
environ["HTTP_PROXY"] = "http://clash.internal.moqi.ai:7890"
environ["HTTPS_PROXY"] = "http://clash.internal.moqi.ai:7890"

from argparse import ArgumentParser
from retrieval_qa_benchmark.models import *
from retrieval_qa_benchmark.datasets import *
from retrieval_qa_benchmark.transforms import *
from retrieval_qa_benchmark.protocols import *
from retrieval_qa_benchmark.utils.registry import REGISTRY
from retrieval_qa_benchmark.utils.profiler import PROFILER
from retrieval_qa_benchmark.utils.config import load

p = ArgumentParser()
p.add_argument('--config', '-c', default=f'')
p.add_argument('--mmlu-subset', '-set', default='prehistory')
p.add_argument('--outdir', '-o', default='results')
p.add_argument('--topk', '-k', default=1)

args = p.parse_args()
config = load(open(args.config))
if 'args' in config['evaluator']['dataset']:
    config['evaluator']['dataset'] = {}
config['evaluator']['dataset']['args'] = {'subset': args.mmlu_subset}
config['evaluator']['transform_chain']['chain'][-1]['args']['num_selected'] = args.topk
print(f"Evaluating MMLU-{config['evaluator']['dataset']['args']['subset']} with "
      f"{config['evaluator']['transform_chain']['chain'][-1]['args']['num_selected']} supporting materials.")

outfile_result = path.join(args.outdir, f"mmlu_{args.mmlu_subset}", f"{args.topk}_m100_p40_gpt35.jsonl")
print('output_file:', outfile_result)

evaluator: MCSAEvaluator = REGISTRY.Evaluators["mcsa"].from_config(config)
# evaluator.dataset.eval_set = evaluator.dataset.eval_set[:5]
acc, matched = evaluator()

avg_token = sum([m.prompt_tokens + m.completion_tokens for m in matched]) / len(matched)

os.makedirs(path.join(args.outdir, f"mmlu_{args.mmlu_subset}"), exist_ok=True)

with open(outfile_result, "w") as f:
    f.write("\n".join([f"Accuracy: {acc:.2f}%", f"Average tokens: {avg_token:.2f}"] + [r.model_dump_json() for r in matched]))
