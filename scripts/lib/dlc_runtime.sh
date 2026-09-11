#!/usr/bin/env bash
# Sourced helpers only. No package installation, controller mutation or launch
# occurs until an entrypoint calls the relevant function explicitly.

nga_load_recipe() {
    local recipe="$1" bindings binding
    bindings="$(PYTHONPATH="$NGA_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$NGA_PYTHON" -m archlab.megatron.launch_config --recipe "$recipe" --bindings)" || return
    while IFS= read -r binding; do
        export "$binding"
    done <<< "$bindings"
    export NGA_LAUNCH_RECIPE="$recipe"
}

nga_require_immutable_repo() {
    test "$(git -C "$NGA_REPO_ROOT" rev-parse HEAD)" = "$NGA_EXPECTED_COMMIT" || return
    local status drift
    status="$(git -C "$NGA_REPO_ROOT" status --porcelain=v1 --untracked-files=all)" || return
    drift="$(printf '%s\n' "$status" | grep -Ev '^\?\? (\.LAUNCH_READY|repo-head\.txt)$' || true)"
    if [[ -n "$drift" ]]; then
        echo "immutable repository is not clean: $drift" >&2
        return 1
    fi
}

nga_container_environment() {
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export PATH="$(dirname "$NGA_PYTHON"):$CUDA_HOME/bin:$PATH"
    export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
    export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}"
    export PYTHONPATH="$NGA_REPO_ROOT/src:$NGA_MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export TOKENIZERS_PARALLELISM=true
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NGA_ALLOW_NONDETERMINISTIC_ALGO:-1}"
    unset NVTE_GROUPED_LINEAR_SINGLE_PARAM
    export NGA_CONTAINER_DIGEST="${NGA_CONTAINER_DIGEST:-sci-agi-zhongwei-registry-vpc.cn-zhongwei.cr.aliyuncs.com/dev/nemo:26.06}"
}

nga_torchrun() {
    "$NGA_PYTHON" -m torch.distributed.run \
        --nnodes="$WORLD_SIZE" \
        --nproc-per-node="$NGA_GPUS_PER_NODE" \
        --node-rank="$RANK" \
        --master-addr="$MASTER_ADDR" \
        --master-port="$MASTER_PORT" \
        --module "$@"
}
