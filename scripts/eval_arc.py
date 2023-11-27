import os
from os import environ, path
from argparse import ArgumentParser
from loguru import logger

from retrieval_qa_benchmark.datasets import *
from retrieval_qa_benchmark.models import *
from retrieval_qa_benchmark.evaluators import *
from retrieval_qa_benchmark.transforms import *
from retrieval_qa_benchmark.utils.config import load
from retrieval_qa_benchmark.schema import BaseEvaluator
from retrieval_qa_benchmark.utils.registry import REGISTRY
from retrieval_qa_benchmark.utils.factory import EvaluatorFactory

p = ArgumentParser()
p.add_argument("--config", "-c", default="../config/arc.yaml")
p.add_argument("--outdir", "-o", default="results")
p.add_argument("--num_retrieval", "-k", default=5, type=int)
p.add_argument("--arc-subset", "-s", default="ARC-Easy", type=str)

args = p.parse_args()
yaml_file = args.config
config = load(open(yaml_file))

assert (
    config["evaluator"]["dataset"]["type"] == "arc"
), "This script is only for evaluating ARC datasets!"
try:
    if "args" not in config["evaluator"]["dataset"]:
        config["evaluator"]["dataset"]["args"] = {}
    config["evaluator"]["dataset"]["args"]["subset"] = args.arc_subset
except Exception as e:
    logger.warning(
        f"{type(e)}: {str(e)} -- Cannot change ARC subset for this evaluation."
    )

k = args.num_retrieval

outfile_result = path.join(
    args.outdir, path.split(args.config)[1] + f"@{args.arc_subset}-{k}.jsonl"
)
print("output_file:", outfile_result)

if os.path.exists(outfile_result):
    Warning(f"File {outfile_result} exists")

if not path.exists(os.path.dirname(outfile_result)):
    os.makedirs(os.path.dirname(outfile_result))

flag = True
while flag:
    try:
        evaluator: BaseEvaluator = EvaluatorFactory.from_config(config).build()
        # evaluator.dataset.eval_set = evaluator.dataset.eval_set[:5]
        if len(evaluator.transform.nodes) > 0:
            try:
                evaluator.transform.nodes["0"].num_selected = k
            except Exception as e:
                logger.warning(f"{str(e)}: Cannot change num retrieval for searcher.")
        acc, matched = evaluator()
        flag = False
    except Exception as e:
        logger.error(f"{str(e)}: failed to build / run evaluator! Retrying...")



avg_token = sum([m.prompt_tokens + m.completion_tokens for m in matched]) / len(matched)

with open(outfile_result, "w") as f:
    f.write(
        "\n".join(
            [f"Accuracy: {acc:.2f}%", f"Average tokens: {avg_token:.2f}"]
            + [r.model_dump_json() for r in matched]
        )
    )
