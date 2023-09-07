from __future__ import annotations

from typing import Any, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


def torch_load_dnn(path: str) -> Any:
    if path.startswith("http:") or path.startswith("https:"):
        dnn = torch.hub.load_state_dict_from_url(path, map_location="cpu")
    else:
        dnn = torch.load(path, map_location="cpu")

    return dnn


class HF_ColBERT(BertPreTrainedModel):
    """
    Shallow wrapper around HuggingFace transformers.
    All new parameters should be defined at this level.

    This makes sure `{from,save}_pretrained` and
    `init_weights` are applied to new parameters correctly.
    """

    _keys_to_ignore_on_load_unexpected = [r"cls"]

    def __init__(self, config: Any, colbert_config: Any):
        super().__init__(config)

        self.config = config
        self.dim = colbert_config.dim
        self.linear = nn.Linear(config.hidden_size, colbert_config.dim, bias=False)
        setattr(self, self.base_model_prefix, BertModel(config))

        # if colbert_config.relu:
        #     self.score_scaler = nn.Linear(1, 1)

        self.init_weights()

        # if colbert_config.relu:
        #     self.score_scaler.weight.data.fill_(1.0)
        #     self.score_scaler.bias.data.fill_(-8.0)

    @property
    def LM(self) -> Any:
        base_model_prefix = getattr(self, "base_model_prefix")
        return getattr(self, base_model_prefix)

    def forward(
        self, *args: Any, **kwargs: Any
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(
            *args, **kwargs
        )
        output.pooler_output = self.linear(output.last_hidden_state)
        return output

    @classmethod
    def from_pretrained(cls, name_or_path: str, colbert_config: Any) -> HF_ColBERT:
        if name_or_path.endswith(".dnn"):
            dnn = torch_load_dnn(name_or_path)
            base = dnn.get("arguments", {}).get("model", "bert-base-uncased")

            obj = super().from_pretrained(
                base, state_dict=dnn["model_state_dict"], colbert_config=colbert_config
            )
            obj.base = base

            return obj

        obj = super().from_pretrained(name_or_path, colbert_config=colbert_config)
        obj.base = name_or_path

        return obj

    @staticmethod
    def raw_tokenizer_from_pretrained(name_or_path: str) -> Any:
        if name_or_path.endswith(".dnn"):
            dnn = torch_load_dnn(name_or_path)
            base = dnn.get("arguments", {}).get("model", "bert-base-uncased")

            obj = AutoTokenizer.from_pretrained(base)
            obj.base = base

            return obj

        obj = AutoTokenizer.from_pretrained(name_or_path)
        obj.base = name_or_path

        return obj
