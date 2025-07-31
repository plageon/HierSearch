# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union

import datasets
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.template import prompt_template_dict


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        self.prompt_template_name = config.get("prompt_template_name", None)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []
        if self.prompt_template_name in prompt_template_dict:
            self.prompt_template = prompt_template_dict[self.prompt_template_name]
        else:
            if "local" in self.prompt_template_name:
                self.prompt_template = {
                    "omnieval": prompt_template_dict["financial_agent_zh_template_sys"],
                    "default": prompt_template_dict["graph_search_agent_template_sys"],
                }
            else:
                self.prompt_template = {
                    "omnieval": prompt_template_dict["web_financial_agent_zh_template_sys"],
                    "default": prompt_template_dict["web_graph_search_agent_template_sys"],
                }

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        # self.prompts = self.dataframe[self.prompt_key]
        # for key in self.prompt_dict_keys:
        #     # type(x): pandas.core.series.Series
        #     # type(x[0]): numpy.ndarray
        #     # type(x[0][0]): dict
        #     try:
        #         self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
        #     except Exception:
        #         print(f"self.prompts={self.prompts}")
        #         raise
        # self.prompts = self.prompts.tolist()
        # self.responses = self.dataframe[self.response_key]
        # for key in self.response_dict_keys:
        #     try:
        #         self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
        #     except Exception:
        #         print(f"self.responses={self.responses}")
        #         raise
        # self.responses = self.responses.tolist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        row_dict: dict = self.dataframe[item]
        prompt = row_dict[self.prompt_key[0]]
        response = row_dict[self.response_key[0]]
        data_source = row_dict.get("data_source", "unknown")

        # apply chat template
        if isinstance(self.prompt_template, str):
            system_prompt = self.prompt_template
        else:
            system_prompt = self.prompt_template.get(self.prompt_template_name, self.prompt_template["default"])
        prompt_chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}]

        # string
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]
        prompt_loss_mask = torch.zeros_like(prompt_attention_mask)

        # response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        # response_ids = response_ids_output["input_ids"][0]
        # response_attention_mask = response_ids_output["attention_mask"][0]

        # mask content between <result> and </result>
        response_ids, response_attention_mask, response_loss_mask = [], [], []
        response_parts = response_chat_str.split("<result>")
        first_part_output = tokenizer(response_parts[0], return_tensors="pt", add_special_tokens=False)
        response_ids.append(first_part_output["input_ids"][0])
        response_attention_mask.append(first_part_output["attention_mask"][0])
        response_loss_mask.append(torch.ones_like(first_part_output["attention_mask"][0]))
        for i in range(1, len(response_parts)):
            assert "</result>" in response_parts[i], f"response_parts={response_parts}"
            result_part = response_parts[i].split("</result>")[0]
            response_part = response_parts[i].split("</result>")[1]
            result_part = f"<result>{result_part}</result>"

            result_part_output = tokenizer(result_part, return_tensors="pt", add_special_tokens=False)
            response_ids.append(result_part_output["input_ids"][0])
            response_attention_mask.append(result_part_output["attention_mask"][0])
            response_loss_mask.append(torch.zeros_like(result_part_output["attention_mask"][0]))

            response_part_output = tokenizer(response_part, return_tensors="pt", add_special_tokens=False)
            response_ids.append(response_part_output["input_ids"][0])
            response_attention_mask.append(response_part_output["attention_mask"][0])
            response_loss_mask.append(torch.ones_like(response_part_output["attention_mask"][0]))

        response_ids = torch.cat(response_ids, dim=-1)
        response_attention_mask = torch.cat(response_attention_mask, dim=-1)
        response_loss_mask = torch.cat(response_loss_mask, dim=-1)

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        # loss_mask = attention_mask.clone()
        # if prompt_length > 1:
        #     # mask out prompt for SFT.
        #     loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
