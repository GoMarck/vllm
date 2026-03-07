# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path

from vllm.model_executor.model_loader.fast_load.meta_types import WeightShardMeta


class FileMetaProvider:
    """File-based metadata provider for the first fast_load version."""

    def __init__(self, meta_path: str):
        path = Path(meta_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._meta_path = path

    def publish(self, model_key: str, meta: WeightShardMeta) -> None:
        data: dict[str, object]
        if self._meta_path.exists():
            with self._meta_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
        data[model_key] = meta.to_dict()
        with self._meta_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)

    def resolve(self, model_key: str) -> WeightShardMeta | None:
        if not self._meta_path.exists():
            return None
        with self._meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        payload = data.get(model_key)
        if payload is None:
            return None
        return WeightShardMeta.from_dict(payload)
