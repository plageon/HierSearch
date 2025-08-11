from typing import List

import numpy as np
import torch

from .Contriever import ContrieverModel
from .base import EmbeddingConfig, BaseEmbeddingModel
from .GritLM import GritLMEmbeddingModel
from .NVEmbedV2 import NVEmbedV2EmbeddingModel
from .OpenAI import OpenAIEmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class STEmbeddingModel:
    def __init__(self, global_config, embedding_model_name) -> None:
        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        import torch
        from sentence_transformers import SentenceTransformer

        self.max_length = global_config.embedding_max_seq_len
        # "float16", "float32", "bfloat16", "auto"
        # if global_config.embedding_model_type == "float16":
        #     torch_dtype = torch.float16
        # elif global_config.embedding_model_type == "float32":
        #     torch_dtype = torch.float
        # elif global_config.embedding_model_type == "bfloat16":
        #     torch_dtype = torch.bfloat16
        # elif global_config.embedding_model_type == "auto":
        #     torch_dtype = torch.float
        self.instruction = ''
        self.normalize_embeddings = global_config.embedding_return_as_normalized
        self.device = global_config.embedding_model_device
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name, trust_remote_code=True, device=self.device, model_kwargs={
                "torch_dtype": global_config.embedding_model_dtype}
        )

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if isinstance(texts, str): texts = [texts]

        params = {
            "prompt": self.instruction,
            "normalize_embeddings": self.normalize_embeddings,
            "show_progress_bar" : False,
        }
        if kwargs: params.update(kwargs)

        if "prompt" in kwargs:
            if kwargs["prompt"] != '':
                params["prompt"] = f"Instruct: {kwargs['prompt']}\nQuery: "
            # del params["prompt"]

        batch_size = params.get("batch_size", 16)

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")
        if len(texts) <= batch_size:
            params["sentences"] = texts  # self._add_eos(texts=texts)
            results = self.embedding_model.encode(**params)
        else:
            # pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                params["sentences"] = texts[i:i + batch_size]
                results.append(self.embedding_model.encode(**params))
                # pbar.update(batch_size)
            # pbar.close()
            # results = torch.cat(results, dim=0)
            results = np.concatenate(results, axis=0)

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()
        if params.get("norm", False):
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results

def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "GritLM" in embedding_model_name:
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        return NVEmbedV2EmbeddingModel
    elif "contriever" in embedding_model_name:
        return ContrieverModel
    elif "text-embedding" in embedding_model_name:
        return OpenAIEmbeddingModel
    else:
        return STEmbeddingModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"