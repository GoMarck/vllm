# vLLM Rapid Start: transfer_engine 首版改造执行计划

## 目标
- 将当前 fast load 流程中的 `dev_mset/dev_mget` 替换为 `transfer_engine`：
  - 发布侧：`batch_register_memory`
  - 拉取侧：`batch_transfer_sync_read`
- 首版远端发现机制：通过环境变量提供的元信息文件实现“后拉起进程发现已加载权重进程”。
- 保持现有兜底：D2D 失败时回退到原始 `load_weights`。

## 模块抽象设计

### 1) 传输引擎适配层（解耦 SDK 依赖）
- 文件：`vllm/model_executor/model_loader/rapid_start/transfer_engine_client.py`
- 职责：
  - 封装 `TransferEngine` 生命周期（init/finalize）
  - 封装 `batch_register_memory` / `batch_transfer_sync_read`
  - 统一错误包装，向上抛 `RuntimeError` 并带 `Status.to_string()`
- 接口草案：
  - `register_tensors(tensors: list[torch.Tensor]) -> list[RegisteredTensorMeta]`
  - `read_into_tensors(target_hostname: str, dst_tensors: list[torch.Tensor], remote_addrs: list[int], sizes: list[int]) -> None`

### 2) 远端元信息抽象层（可替换发现方式）
- 文件：`vllm/model_executor/model_loader/rapid_start/remote_meta_provider.py`
- 职责：定义远端发现抽象，避免写死“文件”方案。
- 接口草案：
  - `class RemoteWeightMetaProvider(Protocol):`
  - `publish(model_key: str, meta: WeightShardMeta) -> None`
  - `resolve(model_key: str) -> WeightShardMeta | None`

### 3) 文件型元信息实现（首版）
- 文件：`vllm/model_executor/model_loader/rapid_start/file_meta_provider.py`
- 职责：
  - 从 `VLLM_RAPID_START_META_FILE` 读写 JSON 元信息
  - 首版只支持单 writer / 后续 reader（满足你描述的流程）
- 元信息内容（按 model_key 分片）：
  - `target_hostname`（`host:port`）
  - `tensor_entries[]`：`name`、`remote_addr`、`size`
  - `world info`：`tp/pp/ep` 维度（用于校验）
  - `dtype/device`（可选，便于调试）

### 4) fast load 编排层
- 文件：`vllm/model_executor/model_loader/rapid_start/fast_load.py`
- 职责：
  - 维持现有入口 `fast_load_weights(model, model_key, load_callback, model_config)`
  - 执行顺序：
    1. `provider.resolve(model_key)`
    2. 若命中：按参数顺序组装 `remote_addrs/sizes`，调用 `batch_transfer_sync_read`
    3. 若未命中或失败：回退 `load_callback`
    4. 回退成功后：调用 `batch_register_memory` + `provider.publish`
- 关键点：参数顺序严格以 `model.named_parameters()` 当前顺序为准，首版使用 `name+size` 双重校验。

### 5) BaseModelLoader 接入
- 文件：`vllm/model_executor/model_loader/base_loader.py`
- 改动：
  - 在 `load_model` 调用 `fast_load_weights(...)`
  - 保留日志：命中/回退/发布耗时

## 环境变量设计（首版）
- `VLLM_RAPID_START_ENABLE`：是否开启 rapid start（默认关）
- `VLLM_RAPID_START_META_FILE`：元信息文件路径（reader/writer 共用）
- `VLLM_RAPID_START_TE_HOSTNAME`：本进程 transfer_engine 初始化地址（`host:port`）
- `VLLM_RAPID_START_TE_DEVICE_ID`：transfer_engine device_id
- `VLLM_RAPID_START_TE_RPC_THREADS`：可选，默认 2

## 代码改造步骤
1. 新增 `rapid_start` 子目录与数据结构定义（meta dataclass + protocol）。
2. 实现 `TransferEngineClient`，先以最小 API 打通注册/同步读取。
3. 实现 `FileMetaProvider`，按 `model_key` 读写 JSON。
4. 实现新的 `fast_load.py` 编排逻辑，替换原 ds 路径。
5. 修改 `base_loader.py` 接入新 fast load。
6. 增加基础单测（无需真实 NPU/TE）：
   - provider 读写正确性
   - fast-load 命中/未命中/失败回退路径
   - 参数数量与 size 不一致时拒绝 D2D
7. 增加最小运行文档到 `ext/rapid_start/README.md`。

## 风险与约束
- `batch_transfer_sync_read` 需要远端进程在线且地址可达；若 owner 退出，reader 必然回退。
- 首版基于参数遍历顺序对齐，若模型定义变更会失配；通过 `name+size` 校验兜底。
- 文件型元信息并发写存在竞态；首版先假设单 writer，后续可引入锁或外部注册中心。

## 验证计划
- 本地 dry-run（mock TransferEngine）验证控制流。
- 双进程联调：
  - 进程 A：正常 load 后注册并写 meta
  - 进程 B：读取 meta，执行 D2D load，不走 `load_callback`
- 故障注入：
  - meta 缺失
  - remote_addrs 数量不匹配
  - transfer_engine 返回错误
  - 均能自动回退 `load_callback`

## 交付物
- 代码：`vllm/model_executor/model_loader/rapid_start/*`
- 接入修改：`vllm/model_executor/model_loader/base_loader.py`
- 文档：`ext/rapid_start/transfer_engine_fast_load_plan.md`、`ext/rapid_start/README.md`
