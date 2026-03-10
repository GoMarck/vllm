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
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )

            logger.debug("Loading weights by rfork on %s ...", load_device)
            logger.info("load_model key: %s", model_key)
            logger.info("DEBUG VALUE| rfork worker is %s", load_config.rfork_worker)
            # Quantization does not happen in `load_weights` but after it
            try:
                if not load_config.rfork_worker.pre_transfer(model):
                    raise RuntimeError("pre_transfer failed.")
                if not load_config.rfork_worker.is_seed_available():
                    raise RuntimeError("seed is not available.")
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
                self.load_config.rfork_worker.post_transfer()
                # Set result: failed.
                self.load_config.rfork_worker.set_transfer_result(False)
                # Cleanup after failed transfer, including unregister RDMA memory regions.
                if (not self.load_config.rfork_worker.cleanup_after_transfer_failed()):
                    raise RuntimeError("cleanup_after_transfer_failed failed.")
                del model
                gc.collect()
                self.load_config.load_format = self.load_config.rfork_fallback_load_format
                logger.info(
                    "fall back into %s to load model",
                    load_config.load_format,
                )
                from vllm.model_executor.model_loader import get_model_loader
                model_loader = get_model_loader(load_config)
                return model_loader.load_model(vllm_config, model_config)
