# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TensorEntryMeta:
    name: str
    remote_addr: int
    size: int
    dtype: str
    shape: list[int]


@dataclass(frozen=True)
class WeightShardMeta:
    target_hostname: str
    tensor_entries: list[TensorEntryMeta]
    model_key: str
    rank_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "WeightShardMeta":
        entries = [
            TensorEntryMeta(
                name=str(entry["name"]),
                remote_addr=int(entry["remote_addr"]),
                size=int(entry["size"]),
                dtype=str(entry["dtype"]),
                shape=[int(dim) for dim in entry["shape"]],
            )
            for entry in data["tensor_entries"]
        ]
        return WeightShardMeta(
            target_hostname=str(data["target_hostname"]),
            tensor_entries=entries,
            model_key=str(data["model_key"]),
            rank_id=str(data["rank_id"]),
        )
