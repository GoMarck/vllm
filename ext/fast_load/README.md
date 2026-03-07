# fast_load (v1)

## Overview
- Replace model weight reuse path from `dev_mset/dev_mget` to `transfer_engine` D2D path.
- First process loads model normally, then registers parameter memory and writes metadata file.
- Later process reads metadata file and uses `batch_transfer_sync_read` to load weights directly.
- If D2D path fails, it always falls back to regular `load_weights`.

## Required Runtime Config
- CLI params:
  - `--tensor-parallel-size` should match between owner/requester.
  - `--pipeline-parallel-size` should match between owner/requester.
  - Other parallel settings (e.g. EP) should match as well.
- Env vars:
  - `VLLM_FAST_LOAD_ENABLE=1`
  - `VLLM_FAST_LOAD_META_FILE=/path/to/meta.json`
  - `VLLM_FAST_LOAD_TE_HOSTNAME=ip:port` (optional, auto infer if unset)
  - `VLLM_FAST_LOAD_TE_RPC_THREADS=2` (optional, default `2`)

## Notes
- Metadata file is automatically split by rank suffix.
- Example input: `/tmp/meta.json`
- Actual file: `/tmp/meta.rank3.json` or `/tmp/meta.ranktp0_pp0_ep0.json`
- Validation before D2D read includes `name/size/dtype/shape`.
- Check logs:
  - hit: `fast_load hit D2D path, ...`
  - fallback: `fast_load D2D path failed, fallback to local loader: ...`
