import os
from os import environ, path
from argparse import ArgumentParser

from retrieval_qa_benchmark.datasets import *
from retrieval_qa_benchmark.models import *
from retrieval_qa_benchmark.evaluators import *
from retrieval_qa_benchmark.transforms import *
from retrieval_qa_benchmark.utils.config import load
from retrieval_qa_benchmark.schema import BaseEvaluator
from retrieval_qa_benchmark.utils.registry import REGISTRY
from retrieval_qa_benchmark.utils.factory import EvaluatorFactory

p = ArgumentParser()
p.add_argument("--config", "-c", default="mmlu-llama2-remote-retrieval-myscale")
p.add_argument("--outdir", "-o", default="results")
p.add_argument("--num_retrieval", "-k", default=5, type=int)

args = p.parse_args()
yaml_file = args.config
config = load(open(yaml_file))

k = 10

outfile_result = path.join(args.outdir, args.config + f"-{k}.jsonl")
print("output_file:", outfile_result)

if os.path.exists(outfile_result):
    Warning(f"File {outfile_result} exists")
    
if not path.exists(os.path.dirname(outfile_result)):
    os.makedirs(os.path.dirname(outfile_result))

evaluator:BaseEvaluator = EvaluatorFactory.from_config(config).build()
evaluator.transform.nodes["0"].num_selected = k
# evaluator.dataset.eval_set = evaluator.dataset.eval_set[:5]
acc, matched = evaluator()

avg_token = sum([m.prompt_tokens + m.completion_tokens for m in matched]) / len(matched)

with open(outfile_result, "w") as f:
    f.write(
        "\n".join(
            [f"Accuracy: {acc:.2f}%", f"Average tokens: {avg_token:.2f}"]
            + [r.model_dump_json() for r in matched]
        )
    )
