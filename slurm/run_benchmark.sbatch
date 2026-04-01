#!/bin/bash
#SBATCH --job-name=spec-decode-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Speculative Decoding Benchmark - Northeastern Cluster
# Usage: sbatch slurm/run_benchmark.sbatch

# echo "=== Job Info ==="
# echo "Job ID: $SLURM_JOB_ID"
# echo "Node: $SLURM_NODELIST"
# echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
# echo "Date: $(date)"
# echo ""

# # Load modules (Northeastern Discovery cluster)
# module load cuda/12.8.0
# module load python/3.13.5

# # Activate virtual environment
# source ~/specdec_env/bin/activate

# Print environment info
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "=== Running All Benchmarks (Qwen + Llama) ==="
python scripts/run_all.py \
    --pairs qwen,llama \
    --K 5 \
    --temperature 1.0 \
    --top-p 1.0 \
    --n-repeats 3 \
    --n-warmup 3 \
    --sweep \
    --output-dir results

echo ""
echo "=== Generating Plots ==="
# Generate plots for each model pair
for PAIR in qwen llama; do
    SWEEP_CSV=$(ls -t results/${PAIR}/sweep_*.csv 2>/dev/null | head -1)
    if [ -n "$SWEEP_CSV" ]; then
        echo "Generating plots for $PAIR..."
        python scripts/generate_plots.py "$SWEEP_CSV" --output-dir "plots/${PAIR}"
    fi
done

echo ""
echo "=== Done ==="
echo "Results: results/{qwen,llama}/"
echo "Plots:   plots/{qwen,llama}/"
