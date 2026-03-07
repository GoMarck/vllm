# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import torch.nn as nn

from vllm.config import ModelConfig
from vllm.distributed.parallel_state import get_ep_group, get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader.fast_load.file_meta_provider import (
    FileMetaProvider,
)
from vllm.model_executor.model_loader.fast_load.meta_types import (
    TensorEntryMeta,
    WeightShardMeta,
)
from vllm.model_executor.model_loader.fast_load.remote_meta_provider import (
    RemoteWeightMetaProvider,
)
from vllm.model_executor.model_loader.fast_load.transfer_engine_client import (
    TransferEngineClient,
)
from torch_npu.npu import current_device

logger = init_logger(__name__)

_global_te_client: TransferEngineClient | None = None
_global_meta_provider: RemoteWeightMetaProvider | None = None

_ENV_ENABLE = "VLLM_FAST_LOAD_ENABLE"
_ENV_META_FILE = "VLLM_FAST_LOAD_META_FILE"
_ENV_TE_HOSTNAME = "VLLM_FAST_LOAD_TE_HOSTNAME"
_ENV_TE_RPC_THREADS = "VLLM_FAST_LOAD_TE_RPC_THREADS"


def _env_enabled(name: str) -> bool:
    val = os.getenv(name)
    if val is None:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _group_rank_and_world(group_getter) -> tuple[int, int]:
    try:
        group = group_getter()
    except Exception:
        return 0, 1
    rank = getattr(group, "rank_in_group", None)
    if rank is None:
        rank = getattr(group, "rank", 0)
    world_size = getattr(group, "world_size", 1)
    return int(rank), int(world_size)


def get_model_key(model_id: str) -> str:
    tp_rank, tp_world_size = _group_rank_and_world(get_tp_group)
    pp_rank, pp_world_size = _group_rank_and_world(get_pp_group)
    ep_rank, ep_world_size = _group_rank_and_world(get_ep_group)
    key = model_id + f"_{tp_rank}_{tp_world_size}_{pp_rank}_{pp_world_size}"
    key += f"_{ep_rank}_{ep_world_size}"
    return key


def _get_rank_id() -> str:
    # Prefer launcher-provided global rank if present.
    rank = os.getenv("RANK")
    if rank:
        return rank
    tp_rank, _ = _group_rank_and_world(get_tp_group)
    pp_rank, _ = _group_rank_and_world(get_pp_group)
    ep_rank, _ = _group_rank_and_world(get_ep_group)
    return f"tp{tp_rank}_pp{pp_rank}_ep{ep_rank}"


def _build_meta_file_path(raw_path: str) -> str:
    path = Path(raw_path)
    rank_id = _get_rank_id()
    if path.suffix:
        return str(path.with_name(f"{path.stem}.rank{rank_id}{path.suffix}"))
    return str(path.with_name(f"{path.name}.rank{rank_id}.json"))


def _get_meta_provider() -> RemoteWeightMetaProvider:
    global _global_meta_provider
    if _global_meta_provider is not None:
        return _global_meta_provider

    raw_path = os.getenv(_ENV_META_FILE)
    if not raw_path:
        raise RuntimeError(f"{_ENV_META_FILE} is required when fast_load is on")
    _global_meta_provider = FileMetaProvider(_build_meta_file_path(raw_path))
    return _global_meta_provider


def _get_transfer_engine_client(
    sample_tensor: "torch.Tensor | None" = None,
) -> TransferEngineClient:
    global _global_te_client
    if _global_te_client is not None:
        return _global_te_client

    local_hostname = TransferEngineClient.infer_local_hostname(
        os.getenv(_ENV_TE_HOSTNAME)
    )
    device_id = current_device()
    rpc_threads = int(os.getenv(_ENV_TE_RPC_THREADS, "2"))
    logger.info(
        "fast_load initializing transfer engine: local_hostname=%s, device_id=%d, "
        "rpc_threads=%d",
        local_hostname,
        device_id,
        rpc_threads,
    )
    _global_te_client = TransferEngineClient(local_hostname, device_id, rpc_threads)
    return _global_te_client


def _collect_named_tensors(model: nn.Module) -> OrderedDict[str, "torch.Tensor"]:
    named_tensors: OrderedDict[str, "torch.Tensor"] = OrderedDict()
    for name, param in model.named_parameters():
        named_tensors[name] = param.data
    return named_tensors


def _validate_meta(named_tensors: OrderedDict[str, "torch.Tensor"],
                   meta: WeightShardMeta) -> None:
    if len(meta.tensor_entries) != len(named_tensors):
        raise RuntimeError(
            "tensor count mismatch, local="
            f"{len(named_tensors)}, remote={len(meta.tensor_entries)}"
        )
    for local_name, local_tensor, remote in zip(
        named_tensors.keys(),
        named_tensors.values(),
        meta.tensor_entries,
        strict=True,
    ):
        if local_name != remote.name:
            raise RuntimeError(
                f"name mismatch, local={local_name}, remote={remote.name}"
            )
        local_size = int(local_tensor.nbytes)
        if local_size != remote.size:
            raise RuntimeError(
                f"size mismatch for {local_name}, local={local_size},"
                f" remote={remote.size}"
            )
        local_dtype = str(local_tensor.dtype)
        if local_dtype != remote.dtype:
            raise RuntimeError(
                f"dtype mismatch for {local_name}, local={local_dtype},"
                f" remote={remote.dtype}"
            )
        local_shape = list(local_tensor.shape)
        if local_shape != remote.shape:
            raise RuntimeError(
                f"shape mismatch for {local_name}, local={local_shape},"
                f" remote={remote.shape}"
            )


def _build_local_meta(model_key: str, named_tensors: OrderedDict[str, "torch.Tensor"],
                      remote_addrs: list[int],
                      target_hostname: str) -> WeightShardMeta:
    entries: list[TensorEntryMeta] = []
    for (name, tensor), remote_addr in zip(
        named_tensors.items(), remote_addrs, strict=True
    ):
        entries.append(
            TensorEntryMeta(
                name=name,
                remote_addr=int(remote_addr),
                size=int(tensor.nbytes),
                dtype=str(tensor.dtype),
                shape=[int(dim) for dim in tensor.shape],
            )
        )
    return WeightShardMeta(
        target_hostname=target_hostname,
        tensor_entries=entries,
        model_key=model_key,
        rank_id=_get_rank_id(),
    )


def fast_load_weights(
    model: nn.Module,
    model_key: str,
    load_callback: Callable[[nn.Module, ModelConfig], None],
    model_config: ModelConfig,
) -> None:
    if not _env_enabled(_ENV_ENABLE):
        load_callback(model, model_config)
        return

    named_tensors = _collect_named_tensors(model)

    try:
        start_time = time.time()
        meta_provider = _get_meta_provider()
        remote_meta = meta_provider.resolve(model_key)
        if remote_meta is not None:
            _validate_meta(named_tensors, remote_meta)
            sample_tensor = next(iter(named_tensors.values()), None)
            te_client = _get_transfer_engine_client(sample_tensor)
            te_client.read_into_tensors(
                remote_meta.target_hostname,
                list(named_tensors.values()),
                [entry.remote_addr for entry in remote_meta.tensor_entries],
                [entry.size for entry in remote_meta.tensor_entries],
            )
            logger.info(
                "fast_load hit D2D path, key=%s, tensors=%d, cost=%.2fs",
                model_key,
                len(named_tensors),
                time.time() - start_time,
            )
            return
    except Exception as e:
        logger.warning("fast_load D2D path failed, fallback to local loader: %s", e)

    logger.info("fast_load miss/fallback path, loading weights from local source")
    load_callback(model, model_config)

    try:
        publish_start = time.time()
        sample_tensor = next(iter(named_tensors.values()), None)
        te_client = _get_transfer_engine_client(sample_tensor)
        remote_addrs = te_client.register_tensors(list(named_tensors.values()))
        local_meta = _build_local_meta(
            model_key,
            named_tensors,
            remote_addrs,
            te_client.local_hostname,
        )
        meta_provider = _get_meta_provider()
        meta_provider.publish(model_key, local_meta)
        logger.info(
            "fast_load published metadata for D2D reuse, key=%s, tensors=%d, "
            "cost=%.2fs",
            model_key,
            len(named_tensors),
            time.time() - publish_start,
        )
    except Exception as e:
        logger.warning("fast_load publish failed: %s", e)
