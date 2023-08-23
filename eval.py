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
p.add_argument('--config', '-c', default=f'mmlu-llama2-remote-retrieval-myscale')

args = p.parse_args()
yaml_file = f'config/{args.config}.yaml'
config = load(open(yaml_file))

outdir = 'results/'

outfile_result = path.join(outdir, args.config + ".jsonl")
print('output_file:', outfile_result)

if os.path.exists(outfile_result):
    Warning(f'File {outfile_result} exists')

evaluator = REGISTRY.Evaluators["mcsa"].from_config(config)
# evaluator.dataset.eval_set = evaluator.dataset.eval_set[:5]
matched = evaluator()

if not path.exists(outdir):
    os.makedirs(outdir)
    
with open(outfile_result, "w") as f:
    f.write("\n".join([f"Accuracy: {matched[0]:.2f}%", f"Average tokens: {matched[1]:.2f}"] + [r.model_dump_json() for r in matched[2]]))
