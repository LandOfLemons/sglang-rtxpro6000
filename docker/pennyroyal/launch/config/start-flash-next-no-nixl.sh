#!/usr/bin/env bash
# Pennyroyal Flash-Next (native NEXTN, no FR-Spec) with GPU radix cache and
# the host-RAM HiCache tier, and WITHOUT the optional NIXL disk tier.
#
# Copy this shape when you want no NIXL mount, no NIXL root and no NIXL
# operational config at all: start it with
#   ../run.sh --startup config/start-flash-next-no-nixl.sh --no-nixl
# --no-nixl leaves the /nixl volume and the io_uring seccomp setting out. An
# existing /nixl directory is never read or deleted by this path, so switching
# back to a NIXL startup script reuses the cache that is already there.
#
# Everything except the three NIXL storage-backend arguments is the same
# launch as config/start-flash-next.sh, so the GPU radix cache and the
# host-RAM cache tier stay enabled; only the disk tier and its namespace
# derivation (which needs the NIXL root) are gone.
set -euo pipefail

# --- Your settings ----------------------------------------------------------
# Quote a value that contains a space.
# Container paths: this directory's host counterpart is mounted at /models.
TARGET_MODEL="/models/RadixArk-Qwen3.8-Flash-Next-NVFP4"
# Host-RAM HiCache tier in decimal GB (SGLang sizes the pool at GB * 1e9).
# This is the cache that stays; it is not the NIXL disk tier.
HICACHE_SIZE_GB=32
# TP1 is the qualified topology; TP_SIZE=2 also needs run.sh --gpu 0,1.
TP_SIZE=1
# RAM keeps the original checkpoint and PLE path; nvme needs the prepared
# snapshot plus run.sh --nvme-ple, which is independent of NIXL.
PENNY_PLE_BACKEND=ram
# cpu keeps vision work off the model GPU; cuda:N needs that extra device.
export SGLANG_MM_PREPROCESS_DEVICE=cpu
# Flash-Next request/state capacity; the qualified pair is 4 requests/24 slots.
MAX_RUNNING_REQUESTS=4
MAX_MAMBA_CACHE_SIZE=24
# ----------------------------------------------------------------------------

export NUMPY_MADVISE_HUGEPAGE=0
export SGLANG_FORWARD_UNKNOWN_TOOLS=true
case "$SGLANG_MM_PREPROCESS_DEVICE" in
  cpu) IMAGE_PROCESSOR_BACKEND=pil ;;
  cuda:*) IMAGE_PROCESSOR_BACKEND=torchvision ;;
  *) echo "Choose SGLANG_MM_PREPROCESS_DEVICE=cpu or cuda:N" >&2; exit 1 ;;
esac

REPO_ROOT="${REPO_ROOT:-/opt/pennyroyal}"
IMAGE_CONFIGS="$REPO_ROOT/configs/pennyroyal"
SGLANG_EXE="${SGLANG_EXE:-$REPO_ROOT/.venv/bin/sglang}"
PYTHON="${PYTHON:-$(dirname "$SGLANG_EXE")/python}"
# /cache is the container mount point run.sh publishes. No NIXL root is
# needed, so nothing here reads or requires NIXL_STORAGE_BASE.
CACHE_BASE="${CACHE_BASE:?Set CACHE_BASE to the durable compiler-cache root}"

source "$IMAGE_CONFIGS/chat-template.sh"
source "$IMAGE_CONFIGS/request-capacity.sh"
source "$IMAGE_CONFIGS/reasoning-effort.sh"
source "$IMAGE_CONFIGS/tp-devices.sh"

CONTEXT_LENGTH=524288
PAGE_SIZE=64
if [[ ! "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "TP_SIZE must be a positive integer" >&2
  exit 1
fi
COMPUTE_DTYPE=bfloat16
KV_DTYPE=fp8_e4m3
MAMBA_SSM_DTYPE=bfloat16
MAMBA_CONV_DTYPE=bfloat16
MAMBA_TRACK_INTERVAL=64
PREFILL_CHUNK_SIZE=4096
if [[ ! "$HICACHE_SIZE_GB" =~ ^[1-9][0-9]*$ ]]; then
  echo "HICACHE_SIZE_GB must be a positive integer number of GB, got '$HICACHE_SIZE_GB'" >&2
  exit 1
fi
for path in "$SGLANG_EXE" "$PYTHON"; do
  [[ -x "$path" ]] || { echo "Required executable missing: $path" >&2; exit 1; }
done
[[ -f "$TARGET_MODEL/config.json" && -f "$TARGET_MODEL/model.safetensors.index.json" ]] || {
  echo "Incomplete target checkpoint: $TARGET_MODEL" >&2
  exit 1
}
# The image entrypoint checks these for its built-in profiles; the exec path
# leaves the roots this script actually uses to the script itself.
[[ -d "$CACHE_BASE" && -w "$CACHE_BASE" ]] || {
  echo "Mount a writable directory at $CACHE_BASE for container UID $(id -u)." >&2
  exit 1
}
mkdir -p "$CACHE_BASE"/{huggingface,torch,torchinductor,triton,cuda,flashinfer,sglang/jit}

# Default to as many consecutive GPUs as TP requires (one scheduler process
# per visible device); an explicit CUDA_VISIBLE_DEVICES still wins.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((TP_SIZE - 1)))"
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
export CC="${CC:-/usr/bin/gcc-15}" CXX="${CXX:-/usr/bin/g++-15}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PENNY_BUILD_JOBS="${PENNY_BUILD_JOBS:-4}"
export MAX_JOBS="${MAX_JOBS:-$PENNY_BUILD_JOBS}" CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$PENNY_BUILD_JOBS}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$PENNY_BUILD_JOBS}"
export FLASHINFER_NINJA_JOBS="${FLASHINFER_NINJA_JOBS:-$PENNY_BUILD_JOBS}" FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-$PENNY_BUILD_JOBS}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_HOME="$CACHE_BASE/huggingface" XDG_CACHE_HOME="$CACHE_BASE"
export TORCH_HOME="$CACHE_BASE/torch" TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torchinductor"
export TRITON_CACHE_DIR="$CACHE_BASE/triton" CUDA_CACHE_PATH="$CACHE_BASE/cuda"
export FLASHINFER_WORKSPACE_BASE="$CACHE_BASE/flashinfer"
export SGLANG_CACHE_DIR="$CACHE_BASE/sglang" SGLANG_JIT_CACHE_DIR="$CACHE_BASE/sglang/jit"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_NUMA_BIND_V2=false SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_MAMBA_CONV_DTYPE="$MAMBA_CONV_DTYPE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

# TP selects a topology, it never grants GPUs: refuse to launch when the
# requested ranks (plus a dedicated cuda:N preprocessor) exceed what is
# visible, instead of letting NCCL fail or the request be ignored.
pennyroyal_check_tp_devices "$TP_SIZE" "$SGLANG_MM_PREPROCESS_DEVICE"

# NVMe preflight imports Torch, Triton, FlashInfer and SGLang. Activate their
# durable cache locations before selecting the optional backend.
source "$IMAGE_CONFIGS/ple-backend.sh"
configure_max_total_tokens

TARGET_OVERRIDES='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}'
printf 'Pennyroyal profile: Flash-Next (native NEXTN, no FR-Spec, no NIXL)\n  runtime: %s\n  target: %s\n  cache root: %s\n  storage backend: none\n' \
  "$SGLANG_EXE" "$TARGET_MODEL" "$CACHE_BASE"

launch_args=(serve \
  --warmups=structured_output \
  --model-path "$TARGET_MODEL" \
  --load-format safetensors \
  --served-model-name pennyroyal \
  --host 0.0.0.0 --port 8001 --tp "$TP_SIZE" \
  --dtype "$COMPUTE_DTYPE" --quantization modelopt_fp4 --kv-cache-dtype "$KV_DTYPE" \
  --mem-fraction-static 0.981 \
  "${TOKEN_CAP_ARGS[@]}" \
  --context-length "$CONTEXT_LENGTH" --json-model-override-args "$TARGET_OVERRIDES" \
  --page-size "$PAGE_SIZE" --max-running-requests "$MAX_RUNNING_REQUESTS" --sleep-on-idle \
  --chunked-prefill-size "$PREFILL_CHUNK_SIZE" \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype "$MAMBA_SSM_DTYPE" \
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" --gdn-mtp-cache-mode none \
  --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer \
  --mamba-track-interval "$MAMBA_TRACK_INTERVAL" \
  --enable-hierarchical-cache --hicache-size "$HICACHE_SIZE_GB" --hicache-host-memory-mode cache \
  --hicache-write-policy write_through --hicache-io-backend kernel \
  --hicache-mem-layout page_first \
  "${PLE_ARGS[@]}" --trust-remote-code \
  --chat-template "$CHAT_TEMPLATE" --image-processor-backend "$IMAGE_PROCESSOR_BACKEND" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-request-time-stats-logging --enable-metrics \
  --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS" \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-draft-model-quantization unquant --watchdog-timeout 1800)
source "$IMAGE_CONFIGS/startup-summary.sh"
pennyroyal_startup_summary "${launch_args[@]}"
exec "$SGLANG_EXE" "${launch_args[@]}"
