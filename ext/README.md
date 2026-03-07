# Fast Load Quick Start

## 目的
- `fast_load` 用于加速 vLLM 模型拉起。
- 首个实例正常加载权重并发布元信息；后续实例通过 D2D 直接读取权重。
- D2D 失败时自动回退到本地权重加载。

## 启动参数（建议）
- `--tensor-parallel-size`: 需要与发布侧保持一致。
- `--pipeline-parallel-size`: 需要与发布侧保持一致。
- 其他并行相关参数（如 EP）: 需要与发布侧保持一致。

说明：
- `fast_load` 首版不新增 vLLM CLI 参数，主要通过环境变量启用。

## 环境变量
- `VLLM_FAST_LOAD_ENABLE=1`
- `VLLM_FAST_LOAD_META_FILE=/path/to/meta.json`
- `VLLM_FAST_LOAD_TE_HOSTNAME=ip:port`（可选，不填则自动推导 IP + 随机端口）
- `VLLM_FAST_LOAD_TE_RPC_THREADS=2`（可选，默认 `2`）

## 双实例示例
```bash
# 实例 A（先启动，发布方）
export VLLM_FAST_LOAD_ENABLE=1
export VLLM_FAST_LOAD_META_FILE=/tmp/vllm_fast_load_meta.json
vllm serve /path/to/model --tensor-parallel-size 1 --pipeline-parallel-size 1
```

```bash
# 实例 B（后启动，读取方）
export VLLM_FAST_LOAD_ENABLE=1
export VLLM_FAST_LOAD_META_FILE=/tmp/vllm_fast_load_meta.json
vllm serve /path/to/model --tensor-parallel-size 1 --pipeline-parallel-size 1
```

## 日志判断是否命中 fast_load
- 命中 D2D:
  - `fast_load hit D2D path, ...`
- 未命中或失败回退:
  - `fast_load D2D path failed, fallback to local loader: ...`
  - `fast_load miss/fallback path, loading weights from local source`
