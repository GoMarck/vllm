# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
import time

import torch
import torch.nn as nn
from torch.nn import Module
import gc

from vllm.model_executor.model_loader.base_loader import BaseModelLoader

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig, LoadFormats
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype
logger = init_logger(__name__)


class RForkModelLoader(BaseModelLoader):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> Module | None:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = self.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        model_key = "model_key"
        with set_default_torch_dtype(model_config.dtype):
            need_del = False
            logger.debug("load_model key is %s, rfork worker is %s", model_key, load_config.rfork_worker)
            try:
                if not load_config.rfork_worker.is_seed_available():
                    raise RuntimeError("seed is not available.")
                with target_device:
                    model = initialize_model(
                        vllm_config=vllm_config, model_config=model_config
                    )
                    need_del = True
                if not load_config.rfork_worker.pre_transfer(model):
                    raise RuntimeError("pre_transfer failed.")
                if not load_config.rfork_worker.transfer(model):
                    raise RuntimeError("transfer failed.")
                if not load_config.rfork_worker.post_transfer():
                    raise RuntimeError("post_transfer failed.")

                # Set result: success.
                self.load_config.rfork_worker.set_transfer_result(True)
                self.load_config.rfork_worker.start_seed_service(model)
                process_weights_after_loading(model, model_config, target_device)
                weight_load_start_time = time.time()
                logger.info(
                    "Loading weights took %.2f seconds",
                    time.time() - weight_load_start_time,
                )
                return model.eval()
            except Exception as e:
                logger.exception("RFork transfer failed, cleaning up and falling back: %s", e)
                self.load_config.rfork_worker.post_transfer()
                self.load_config.rfork_worker.set_transfer_result(False)
                if need_del:
                    del model
                    gc.collect()
                    torch.npu.empty_cache()
                    for _ in range(3):
                        gc.collect()
                        torch.npu.empty_cache()
                
                vllm_config.compilation_config.static_forward_context.clear()
                self.load_config.load_format = self.load_config.rfork_fallback_load_format
                logger.info(
                    "fall back into %s to load model",
                    load_config.load_format,
                )
                from vllm.model_executor.model_loader import get_model
                return get_model(vllm_config=vllm_config)