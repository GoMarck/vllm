# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, get_open_port, join_host_port

logger = init_logger(__name__)

if TYPE_CHECKING:
    import torch


class TransferEngineClient:
    """Thin wrapper around transfer_engine Python bindings."""

    def __init__(self, local_hostname: str, device_id: int, rpc_threads: int):
        from transfer_engine import TransferEngine

        self.local_hostname = local_hostname
        self._engine = TransferEngine()
        status = self._engine.initialize(local_hostname, device_id, rpc_threads)
        if status.is_error():
            raise RuntimeError(
                "TransferEngine initialize failed: " + status.to_string()
            )

    @staticmethod
    def infer_local_hostname(env_hostname: str | None) -> str:
        if env_hostname:
            return env_hostname
        return join_host_port(get_ip(), get_open_port())

    @staticmethod
    def infer_device_id_from_tensor(
        tensor: "torch.Tensor | None" = None,
    ) -> int:
        logical_device_id: int | None = None
        if tensor is not None:
            try:
                if tensor.device.index is not None:
                    logical_device_id = int(tensor.device.index)
            except Exception:
                pass

        if logical_device_id is None:
            try:
                import torch
            except Exception:
                logger.warning("Cannot import torch, fallback fast_load device_id=0")
                return 0

            try:
                if hasattr(torch, "npu") and torch.npu.is_available():
                    logical_device_id = int(torch.npu.current_device())
            except Exception:
                pass
            try:
                if logical_device_id is None and torch.cuda.is_available():
                    logical_device_id = int(torch.cuda.current_device())
            except Exception:
                pass
            try:
                if (logical_device_id is None and hasattr(torch, "xpu")
                        and torch.xpu.is_available()):
                    logical_device_id = int(torch.xpu.current_device())
            except Exception:
                pass

        if logical_device_id is None:
            logger.warning("Cannot infer active device, fallback fast_load device_id=0")
            return 0

        # Map logical device id (inside process) to physical id if runtime
        # masks devices via *_VISIBLE_DEVICES.
        env_names = (
            "ASCEND_RT_VISIBLE_DEVICES",
            "ASCEND_VISIBLE_DEVICES",
            "NPU_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        )
        for env_name in env_names:
            raw = os.getenv(env_name, "").strip()
            if not raw:
                continue
            visible = [item.strip() for item in raw.split(",") if item.strip()]
            if logical_device_id < 0 or logical_device_id >= len(visible):
                logger.warning(
                    "fast_load device mapping skipped: %s=%s, logical_id=%d out "
                    "of range",
                    env_name,
                    raw,
                    logical_device_id,
                )
                return logical_device_id
            try:
                mapped = int(visible[logical_device_id])
            except ValueError:
                logger.warning(
                    "fast_load device mapping uses logical id because %s entry is "
                    "non-numeric: %s",
                    env_name,
                    visible[logical_device_id],
                )
                return logical_device_id
            logger.info(
                "fast_load mapped logical device id %d to physical id %d via %s=%s",
                logical_device_id,
                mapped,
                env_name,
                raw,
            )
            return mapped

        return logical_device_id

    def register_tensors(self, tensors: list["torch.Tensor"]) -> list[int]:
        addrs = [int(tensor.data_ptr()) for tensor in tensors]
        sizes = [int(tensor.nbytes) for tensor in tensors]
        status = self._engine.batch_register_memory(addrs, sizes)
        if status.is_error():
            raise RuntimeError(
                "batch_register_memory failed: " + status.to_string()
            )
        return addrs

    def read_into_tensors(
        self,
        target_hostname: str,
        dst_tensors: list["torch.Tensor"],
        remote_addrs: list[int],
        sizes: list[int],
    ) -> None:
        dst_addrs = [int(tensor.data_ptr()) for tensor in dst_tensors]
        status = self._engine.batch_transfer_sync_read(
            target_hostname,
            dst_addrs,
            remote_addrs,
            sizes,
        )
        if status.is_error():
            raise RuntimeError(
                "batch_transfer_sync_read failed: " + status.to_string()
            )

    def finalize(self) -> None:
        status = self._engine.finalize()
        if status.is_error():
            logger.warning("TransferEngine finalize returned error: %s",
                           status.to_string())
