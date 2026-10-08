#!/usr/bin/env bash
# Train every (model, seed) pair on a multi-GPU machine with plain Docker, then run
# the comparison, noise ablation and curvature analysis.
#
# Usage (from the repository root on the DGX):
#   GPUS="0 1 2 3" JOBS_PER_GPU=2 scripts/dgx_train.sh
#
# Environment variables:
#   GPUS          GPU indices you are allowed to use (default: "0"). On a shared
#                 machine, check `nvidia-smi` and only list free GPUs.
#   JOBS_PER_GPU  Concurrent trainings per GPU (default: 2). These models are small
#                 and partly data-loader bound, so two runs per GPU improve utilisation.
#   SEEDS         Training seeds (default: "0 1 2").
#   MODELS        Models to train (default: all five).
#   IMAGE         Docker image tag (default: molpred).
#
# Every run is skipped if its results already exist, so the script can be re-launched
# after an interruption. Logs go to outputs/logs/.

set -euo pipefail

GPUS=(${GPUS:-0})
JOBS_PER_GPU=${JOBS_PER_GPU:-2}
SEEDS=(${SEEDS:-0 1 2})
MODELS=(${MODELS:-GIN GINEdge GINDist EGNNStatic EGNN})
IMAGE=${IMAGE:-molpred}

mkdir -p data outputs/logs results

run_container() {
    # $1: GPU index; remaining arguments are passed to main.py.
    local gpu=$1
    shift
    docker run --rm \
        --gpus "device=${gpu}" \
        --shm-size=16g \
        --user "$(id -u):$(id -g)" \
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

for command in compare noise curvature; do
    echo "Running ${command}..."
    run_container "${GPUS[0]}" "$command" --models "${MODELS[@]}" --seeds "${SEEDS[@]}" \
        --device cuda 2>&1 | tee "outputs/logs/${command}.log"
done

echo "Done. Results and figures are in results/ (commit them to the repository)."
