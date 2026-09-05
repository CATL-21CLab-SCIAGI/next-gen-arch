#!/usr/bin/env bash
set -euo pipefail

# Compatibility entrypoint for resident controllers whose launcher allowlist
# was frozen when the DLC allocation started. Historical 27B runs remain
# reproducible from their pinned commits; at this commit this path accepts only
# one of the exact Flash-Next compatibility contracts and forwards to the
# container-native launcher.
: "${NGA_OUTPUT_ROOT:?set the validated controller compatibility handle}"
: "${NGA_EXPECTED_COMMIT:?set the immutable repository commit}"

if [[ "${NGA_EXPECTED_NODES:-}" != "4" || "${NGA_GPUS_PER_NODE:-}" != "8" ]]; then
    echo "compatibility launch requires the 4-node, 8-GPU-per-node allocation" >&2
    exit 1
fi
if [[ "${NGA_SEQUENCE_LENGTH:-}" != "2048" ]]; then
    echo "compatibility launch requires sequence length 2048" >&2
    exit 1
fi
if [[ "${NGA_GLOBAL_BATCH_SIZE:-}" != "4096" ]]; then
    echo "compatibility launch requires global batch 4096" >&2
    exit 1
fi
if [[ "${NGA_TARGET_TRAIN_TOKENS:-}" != "100000595968" ]]; then
    echo "compatibility launch requires exactly 100000595968 effective tokens" >&2
    exit 1
fi
if [[ "${NGA_CHECKPOINT_INTERVAL_TOKENS:-}" != "100000595968" ]]; then
    echo "legacy validation interval must equal the target; the trainer saves every 1192 steps" >&2
    exit 1
fi
case "$NGA_OUTPUT_ROOT" in
    /mnt/nas/evergreen/next-gen-arch/compat-qwen38-flash-next-*|\
    /mnt/nas/evergreen/compat-qwen38-flash-next-*) ;;
    *)
        echo "invalid compatibility output handle" >&2
        exit 1
        ;;
esac

production_name="${NGA_OUTPUT_ROOT##*/compat-}"
export NGA_OUTPUT_ROOT="/mnt/oss/evergreen/next-gen-arch/$production_name"
NGA_PROBE_STEPS="${NGA_PROBE_STEPS:-0}"
if ((NGA_PROBE_STEPS > 0)); then
    case "$production_name" in
        *-probe-*) ;;
        *)
            echo "probe generations require a probe-labelled output handle" >&2
            exit 1
            ;;
    esac
    export NGA_PROBE_SAVE_INTERVAL=1
fi
export NGA_PROBE_STEPS
case "$production_name" in
    qwen38-flash-next-1b-depth48-nomtp-*)
        export NGA_FLASH_NEXT_MODEL_VARIANT=1b-depth48-no-mtp
        ;;
    qwen38-flash-next-quarter-depth48-nomtp-*)
        export NGA_FLASH_NEXT_MODEL_VARIANT=quarter-depth48-no-mtp
        ;;
    qwen38-flash-next-dense-ple-*)
        export NGA_FLASH_NEXT_MODEL_VARIANT=full
        ;;
    *)
        echo "compatibility handle does not identify a supported Flash-Next variant" >&2
        exit 1
        ;;
esac

exec bash "$(dirname "$0")/run_qwen38_flash_next_full_dlc.sh"
