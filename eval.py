# ruff: noqa: F403, E402
from os import environ, path

# environ['HF_HOME'] = '/Volumes/DATA/hf_home'
environ["HF_HOME"] = "/mnt/workspaces/hf_home"
environ["HTTP_PROXY"] = "http://clash.internal.moqi.ai:7890"
environ["HTTPS_PROXY"] = "http://clash.internal.moqi.ai:7890"
environ["OPENAI_API_BASE"] = "http://10.1.3.28:8990/v1"
environ["OPENAI_API_KEY"] = "sk-qB3QC4y6OtIDASTcE10c06A9F24c462eA55f95867920Df8e"

from argparse import ArgumentParser

import yaml

from retrieval_qa_benchmark.datasets import *
from retrieval_qa_benchmark.models import *
from retrieval_qa_benchmark.protocols import *
from retrieval_qa_benchmark.transforms import *
from retrieval_qa_benchmark.utils.profiler import PROFILER
from retrieval_qa_benchmark.utils.registry import REGISTRY

p = ArgumentParser()
p.add_argument("--config", "-c", default="config/exp.yaml")
p.add_argument("--outdir", "-o", default="results")

args = p.parse_args()
config = yaml.safe_load(open(args.config))

try:
    _args = config["evaluator"]["transform_chain"][1]["args"]
    rank_list = ["m" + str(_args["num_selected"]), str(_args["num_filtered"])]
    for k, v in _args["rank_dict"].items():
        rank_list.append(k[0] + str(v))
except Exception:
    rank_list = ["base"]
rank_list.append("llama")
output_name = "_".join(rank_list)
print("output_name:", output_name)

evaluator = REGISTRY.Evaluators["mcsa"].from_config(config)
# evaluator.dataset.eval_set = evaluator.dataset.eval_set[:5]
matched = evaluator()

outfile_result = path.join(args.outdir, output_name + ".jsonl")
outfile_profile = path.join(args.outdir, output_name + ".txt")

with open(outfile_result, "w") as f:
    f.write("\n".join([r.model_dump_json() for r in matched[1]]))

with open(outfile_profile, "w") as f:
    f.write(str(PROFILER))

print("output_name:", output_name)
