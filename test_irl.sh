export RAY_TMPDIR=/home/j84403411/ray_tmp
mkdir -p "$RAY_TMPDIR"

JOB_ID=${SLURM_JOB_ID:-0}
# Keep ports comfortably low and below 65535
BASE=$(( (JOB_ID % 200) * 50 )) 

export RAY_PORT=$((2000 + BASE))
export RAY_DASHBOARD_PORT=$((6000 + BASE))
export RAY_GCS_PORT=$((7000 + BASE))
export RAY_NODE_MGR_PORT=$((8000 + BASE))
export RAY_OBJ_MGR_PORT=$((9000 + BASE))
export RAY_MIN_WORKER_PORT=$((10000 + BASE))
export RAY_MAX_WORKER_PORT=$((10099 + BASE))

source openpi-venv/bin/activate

ray stop --force >/dev/null 2>&1

IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}')

export RAY_ADDRESS="${IP}:${RAY_PORT}"

ray start --head \
  --port=$RAY_PORT \
  --dashboard-port=$RAY_DASHBOARD_PORT \
  --temp-dir="$RAY_TMPDIR" \
  --num-cpus=$SLURM_CPUS_ON_NODE \
  --disable-usage-stats

bash examples/embodiment/run_embodiment_irl.sh libero_object_ppo_openpi_pi05_irl
