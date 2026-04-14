#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --gres=gpu:1
#SBATCH --partition=agent-xlong
#SBATCH --time=100:00:00
#SBATCH --output=slurm_logs/slurm_%j.out

export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_tmp_${USER}}
mkdir -p "$RAY_TMPDIR"
chmod 700 "$RAY_TMPDIR" 2>/dev/null || true

JOB_ID=${SLURM_JOB_ID:-0}
# Keep ports comfortably low and below 65535
BASE=$(( (JOB_ID % 200) * 20 ))

export RAY_PORT=$((2000 + BASE))
export RAY_DASHBOARD_PORT=$((6000 + BASE))
export RAY_GCS_PORT=$((7000 + BASE))
export RAY_NODE_MGR_PORT=$((8000 + BASE))
export RAY_OBJ_MGR_PORT=$((9000 + BASE))
export RAY_MIN_WORKER_PORT=$((10000 + BASE))
export RAY_MAX_WORKER_PORT=$((10099 + BASE))

source openpi_venv/bin/activate

ray stop --force >/dev/null 2>&1

# Fall back when running outside Slurm
NUM_CPUS=${SLURM_CPUS_ON_NODE:-$(getconf _NPROCESSORS_ONLN 2>/dev/null)}
NUM_CPUS=${NUM_CPUS:-1}

# Avoid netlink permission issues in restricted environments.
# 127.0.0.1 works for single-node local runs.
IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}')
if [ -z "$IP" ]; then
  IP="127.0.0.1"
fi

export RAY_ADDRESS="${IP}:${RAY_PORT}"

ray start --head \
  --node-ip-address="$IP" \
  --port=$RAY_PORT \
  --dashboard-port=$RAY_DASHBOARD_PORT \
  --temp-dir="$RAY_TMPDIR" \
  --num-cpus="$NUM_CPUS" \
  --disable-usage-stats

bash examples/sft/run_embodiment_disc.sh libero_sft_openpi05_object_irl_disc