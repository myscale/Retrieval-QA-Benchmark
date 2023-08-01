from __future__ import annotations

from typing import Dict, Any
from hashlib import sha256
from retrieval_qa_benchmark.datasets.base import HFDataset, build_hfdataset_internal
from retrieval_qa_benchmark.utils.transforms import BaseTransform


def build_id(d: Dict[str, Any], qkey: str = "question", ckey: str = "choices") -> str:
    return sha256((d[qkey] + "".join(d[ckey])).encode("utf-8")).hexdigest()


def build_answer(d: Dict[str, Any], akey: str = "answer", ckey: str = "choices") -> str:
    return f"{chr(65 + d[akey])}. {d[ckey][d[akey]]}"


def build_question(
    d: Dict[str, Any],
    qkey: str = "question",
    ckey: str = "choices",
) -> str:
    question = d[qkey]
    choices = "\t".join([f"{chr(65+i)}. {v}" for i, v in enumerate(d[ckey])])
    return f"{question}\n{choices}"


class MMLU(HFDataset):
    """https://huggingface.co/datasets/hotpot_qa
    Hotpot QA Dataset from Huggingface
    """

    @classmethod
    def build(cls, subset: str = "prehistory") -> MMLU:
        transform = BaseTransform()
        transform.set_value_function("id", build_id)
        transform.set_value_function("answer", build_answer)
        transform.set_value_function("question", build_question)
        transform.set_value_function("type", lambda x: "MCSA")
        name, eval_set = build_hfdataset_internal(
            name=["cais/mmlu", subset],
            eval_split="test",
            transform=transform,
        )
        return cls(name=name, eval_set=eval_set)
