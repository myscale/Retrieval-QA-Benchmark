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
config = load(open("config/mmlu-llama2-remote-retrieval-myscale.yaml"))

try:
    subset = 'mmlu_' + config['evaluator']['dataset']['args']['subset']
except Exception as e:
    subset = 'mmlu_prehistory'
outdir = f'results/{subset}'

try:
    args1 = config['evaluator']['transform_chain']['chain'][0]['args']
    args2 = config['evaluator']['transform_chain']['chain'][1]['args']
    rank_list = [str(args2['num_selected']), 'f' +str(args1['num_selected'])]
    for k,v in args2['rank_dict'].items():
        rank_list.append(k[0]+str(v))
except Exception as e:
    rank_list = ['0']
rank_list.append('llama')
output_name = '_'.join(rank_list)
outfile_result = path.join(outdir, output_name + ".jsonl")
outfile_profile = path.join(outdir, output_name + ".txt")

print('')
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
    
with open(outfile_profile, 'w') as f:
    f.write(str(PROFILER))
