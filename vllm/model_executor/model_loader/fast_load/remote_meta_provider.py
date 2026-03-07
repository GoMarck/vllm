# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Protocol

from vllm.model_executor.model_loader.fast_load.meta_types import WeightShardMeta


class RemoteWeightMetaProvider(Protocol):
    def publish(self, model_key: str, meta: WeightShardMeta) -> None:
        ...

    def resolve(self, model_key: str) -> WeightShardMeta | None:
        ...
