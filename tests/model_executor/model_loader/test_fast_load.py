# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.model_executor.model_loader.fast_load.fast_load import fast_load_weights
from vllm.model_executor.model_loader.fast_load.meta_types import (
    TensorEntryMeta,
    WeightShardMeta,
)


def _build_meta_for_model(model: nn.Module, model_key: str) -> WeightShardMeta:
    entries = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        entries.append(
            TensorEntryMeta(
                name=name,
                remote_addr=1000 + idx,
                size=int(param.data.nbytes),
                dtype=str(param.data.dtype),
                shape=[int(dim) for dim in param.data.shape],
            )
        )
    return WeightShardMeta(
        target_hostname="127.0.0.1:19090",
        tensor_entries=entries,
        model_key=model_key,
        rank_id="0",
    )


def test_fast_load_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_FAST_LOAD_ENABLE", raising=False)
    model = nn.Linear(4, 3)
    calls = {"load": 0}

    def load_cb(_model, _model_config):
        calls["load"] += 1

    fast_load_weights(model, "k0", load_cb, object())  # type: ignore[arg-type]
    assert calls["load"] == 1


def test_fast_load_hit_uses_d2d(monkeypatch):
    monkeypatch.setenv("VLLM_FAST_LOAD_ENABLE", "1")
    model = nn.Linear(4, 3)
    meta = _build_meta_for_model(model, "k1")
    calls = {"load": 0, "read": 0}

    class FakeProvider:
        def resolve(self, model_key):
            assert model_key == "k1"
            return meta

        def publish(self, model_key, payload):
            raise AssertionError(f"publish should not run: {model_key}, {payload}")

    class FakeClient:
        local_hostname = "127.0.0.1:19999"

        def read_into_tensors(self, target_hostname, dst_tensors, remote_addrs, sizes):
            assert target_hostname == meta.target_hostname
            assert len(dst_tensors) == len(meta.tensor_entries)
            assert remote_addrs == [entry.remote_addr for entry in meta.tensor_entries]
            assert sizes == [entry.size for entry in meta.tensor_entries]
            calls["read"] += 1

        def register_tensors(self, _tensors):
            raise AssertionError("register_tensors should not run on D2D hit")

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.fast_load.fast_load._get_meta_provider",
        lambda: FakeProvider(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.fast_load.fast_load._get_transfer_engine_client",  # noqa: E501
        lambda: FakeClient(),
    )

    def load_cb(_model, _model_config):
        calls["load"] += 1

    fast_load_weights(model, "k1", load_cb, object())  # type: ignore[arg-type]
    assert calls["read"] == 1
    assert calls["load"] == 0


def test_fast_load_miss_fallback_and_publish(monkeypatch):
    monkeypatch.setenv("VLLM_FAST_LOAD_ENABLE", "1")
    model = nn.Linear(4, 3)
    calls = {"load": 0, "publish": 0}

    class FakeProvider:
        def resolve(self, model_key):
            assert model_key == "k2"
            return None

        def publish(self, model_key, payload):
            assert model_key == "k2"
            assert len(payload.tensor_entries) > 0
            calls["publish"] += 1

    class FakeClient:
        local_hostname = "127.0.0.1:19999"

        def read_into_tensors(self, target_hostname, dst_tensors, remote_addrs, sizes):
            raise AssertionError(
                f"read should not run on miss: {target_hostname}, {dst_tensors}, "
                f"{remote_addrs}, {sizes}"
            )

        def register_tensors(self, tensors):
            return [int(t.data_ptr()) for t in tensors]

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.fast_load.fast_load._get_meta_provider",
        lambda: FakeProvider(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.fast_load.fast_load._get_transfer_engine_client",  # noqa: E501
        lambda: FakeClient(),
    )

    def load_cb(_model, _model_config):
        for _name, param in model.named_parameters():
            param.data.copy_(torch.ones_like(param.data))
        calls["load"] += 1

    fast_load_weights(model, "k2", load_cb, object())  # type: ignore[arg-type]
    assert calls["load"] == 1
    assert calls["publish"] == 1
