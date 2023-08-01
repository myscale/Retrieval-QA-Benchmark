from __future__ import annotations

from typing import Dict, Any
from hashlib import sha256
from retrieval_qa_benchmark.datasets.base import HFDataset, build_hfdataset_internal
from retrieval_qa_benchmark.utils.transforms import BaseTransform


def build_answer(
    d: Dict[str, Any], akey: str = "answer", ckey: str = "candidates"
) -> str:
    return f"{chr(65 + d[ckey].index(d[akey]))} {d[akey]}"


def build_question(
    d: Dict[str, Any],
    qkey: str = "question",
    ckey: str = "candidates",
) -> str:
    # (TODO: @mpskex) question need to be revised
    question = d[qkey]
    choices = "\t".join([f"{chr(65+i)}. {v}." for i, v in enumerate(d[ckey])])
    return f"{question}\n{choices}"


class WikiHop(HFDataset):
    """https://huggingface.co/datasets/hotpot_qa
    Hotpot QA Dataset from Huggingface
    """

    @classmethod
    def build(cls, subset: str = "original") -> WikiHop:
        transform = BaseTransform(
            value_functions={
                "id": lambda x: x["id"],
                "answer": build_answer,
                "question": build_question,
                "type": lambda x: "MCSA",
            }
        )
        name, eval_set = build_hfdataset_internal(
            name=["wiki_hop", subset],
            eval_split="validation",
            transform=transform,
        )
        return cls(name=name, eval_set=eval_set)
