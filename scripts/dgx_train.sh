#!/usr/bin/env bash
# Train every (model, seed) pair on a multi-GPU machine with plain Docker, then run
# the comparison, noise ablation and curvature analysis.
#
# Usage (from the repository root on the DGX):
#   scripts/dgx_train.sh                              # auto-detect your devices
#   GPUS="0 1 2 3" JOBS_PER_GPU=2 scripts/dgx_train.sh
#
# Environment variables:
#   GPUS          GPU indices or MIG UUIDs to use. Default: every MIG partition shown
#                 by `nvidia-smi -L` (detected at each launch, since partitions can
#                 change between sessions), or "0" if the GPUs are not partitioned.
#   JOBS_PER_GPU  Concurrent trainings per device. Default: 1 on MIG partitions, 2 on
#                 full GPUs (the models are small and partly data-loader bound).
#   SEEDS         Training seeds (default: "0 1 2").
#   MODELS        Runs to train (default: the five architectures). A run can add
#                 training-time coordinate noise: MODELS="GINDist+noise0.05 EGNN+noise0.05".
#   IMAGE         Docker image tag (default: molpred).
#
# Every run is skipped if its results already exist, so the script can be re-launched
# after an interruption. Logs go to outputs/logs/.

set -euo pipefail

# Default devices: the MIG partitions visible to you (they can change between
# sessions, so detect them every time), or GPU 0 if the GPUs are not partitioned.
MIG_UUIDS=$(nvidia-smi -L 2>/dev/null | grep -o 'MIG-[0-9a-f-]*' || true)
if [[ -n "$MIG_UUIDS" ]]; then
    GPUS=(${GPUS:-$MIG_UUIDS})
    JOBS_PER_GPU=${JOBS_PER_GPU:-1}  # MIG slices are small: one training each
else
    GPUS=(${GPUS:-0})
    JOBS_PER_GPU=${JOBS_PER_GPU:-2}
fi
echo "Devices: ${GPUS[*]} (${JOBS_PER_GPU} job(s) each)"
SEEDS=(${SEEDS:-0 1 2})
MODELS=(${MODELS:-GIN GINEdge GINDist EGNNStatic EGNN})
IMAGE=${IMAGE:-molpred}

mkdir -p data outputs/logs results

# Rootless Docker maps container root to the calling user, so files written to the
# bind mounts already belong to you, and `--user` would fail for UIDs outside the
# subuid range ("setgroups: invalid argument"). With rootful Docker, run as yourself
# so the outputs are not owned by root.
if docker info --format '{{.SecurityOptions}}' 2>/dev/null | grep -q rootless; then
    USER_FLAGS=()
else
    USER_FLAGS=(--user "$(id -u):$(id -g)")
fi

run_container() {
    # $1: GPU index or MIG UUID; remaining arguments are passed to main.py.
    local gpu=$1
    shift
    docker run --rm \
        --gpus "device=${gpu}" \
        --shm-size=16g \
        ${USER_FLAGS[@]+"${USER_FLAGS[@]}"} \
        -e HOME=/tmp \
        -v "$PWD/data:/app/data" \
        -v "$PWD/outputs:/app/outputs" \
        -v "$PWD/results:/app/results" \
        "$IMAGE" "$@"
}

echo "Building image ${IMAGE}..."
docker build -t "$IMAGE" .

# Download and process QM9 once, before parallel jobs could race on it.
run_container "${GPUS[0]}" data --device cuda

slots=$(( ${#GPUS[@]} * JOBS_PER_GPU ))
job=0
for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # Wait for a free slot.
        while (( $(jobs -rp | wc -l) >= slots )); do
            wait -n
        done
        gpu=${GPUS[$(( job % ${#GPUS[@]} ))]}
        log="outputs/logs/${model}_seed${seed}.log"
        echo "[$(date +%H:%M:%S)] ${model} seed ${seed} -> GPU ${gpu} (log: ${log})"
        run_container "$gpu" train --models "$model" --seeds "$seed" --device cuda \
            > "$log" 2>&1 &
        job=$(( job + 1 ))
    done
done
wait
echo "All trainings finished."

# The analyses cover every finished run (not only the ones trained now), so the
# noise-trained variants are compared with the original models in one table.
for command in compare noise curvature; do
    echo "Running ${command}..."
    run_container "${GPUS[0]}" "$command" --device cuda 2>&1 | tee "outputs/logs/${command}.log"
done

echo "Done. Results and figures are in results/ (commit them to the repository)."
