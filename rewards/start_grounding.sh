#!/bin/bash

set -e

PYTHON_BIN="${PYTHON_BIN:-python3}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

GPU_IDS_STR="${1:-0,1,2,3}"

IFS=',' read -ra GPU_IDS_ARR <<< "$GPU_IDS_STR"
NUM_GPUS=${#GPU_IDS_ARR[@]}

DINO_PORT=12343

# GroundingDINO worker port list
DINO_WORKER_PORTS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    DINO_WORKER_PORTS+=($((12344 + i)))
done

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ─────────────────────────────────────────────
# Start GroundingDINO (N workers + LB)
# ─────────────────────────────────────────────
echo -e "${YELLOW}Starting GroundingDINO service (${NUM_GPUS} workers + LB)...${NC}"
echo "  Load balancer port: $DINO_PORT"
echo "  Worker ports:       ${DINO_WORKER_PORTS[*]}"
echo "  Worker GPU IDs:     ${GPU_IDS_ARR[*]}"
echo ""

cd reward_server

# Launch N workers in parallel, each bound to its own GPU
DINO_WORKER_PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    port=${DINO_WORKER_PORTS[$i]}
    gpu=${GPU_IDS_ARR[$i]}
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON_BIN" groundingdino_server.py --port $port \
        > ../groundingdino_worker_${TIMESTAMP}_${i}.log 2>&1 &
    DINO_WORKER_PIDS+=($!)
    echo "  Worker $i — GPU $gpu, port $port, PID ${DINO_WORKER_PIDS[$i]}"
done

# Start the load balancer (aggregates all worker backends)
BACKENDS_ARG=$(IFS=,; echo "${DINO_WORKER_PORTS[*]/#/http://localhost:}")
nohup "$PYTHON_BIN" groundingdino_lb.py --port $DINO_PORT --backends "$BACKENDS_ARG" \
    > ../groundingdino_lb_${TIMESTAMP}.log 2>&1 &
DINO_LB_PID=$!
echo "  Load Balancer — port $DINO_PORT, PID $DINO_LB_PID"

cd ..

echo ""
echo "  Waiting for all GroundingDINO workers to finish loading (each takes about 30-60 seconds)..."
echo -n "  ["

DINO_READY=false
for i in {1..200}; do
    sleep 3

    # Check whether all worker processes are still alive
    for j in $(seq 0 $((NUM_GPUS - 1))); do
        if ! ps -p ${DINO_WORKER_PIDS[$j]} > /dev/null; then
            echo -e "${RED}]${NC}"
            echo -e "${RED}Error: GroundingDINO worker $j (PID ${DINO_WORKER_PIDS[$j]}) has exited${NC}"
            echo "Check the log: tail -f groundingdino_worker_${j}.log"
            exit 1
        fi
    done

    # Use the LB's /health endpoint to check whether all workers are ready
    if curl -s "http://localhost:$DINO_PORT/health" | grep -q '"model_loaded":true'; then
        echo -e "${GREEN}✓] All workers started successfully!${NC}"
        DINO_READY=true
        break
    fi

    echo -n "."
done

echo ""

if [ "$DINO_READY" != "true" ]; then
    echo -e "${RED}Error: GroundingDINO service startup timed out${NC}"
    echo "Check the logs:"
    for j in $(seq 0 $((NUM_GPUS - 1))); do
        echo "  tail -f groundingdino_worker_${j}.log"
    done
    exit 1
fi

echo -e "${GREEN}✓ GroundingDINO service is ready (${NUM_GPUS} workers, LB on :$DINO_PORT)${NC}"
echo "  Log files: groundingdino_lb.log, groundingdino_worker_0.log ..."
echo ""

# ─────────────────────────────────────────────
# Summary output
# ─────────────────────────────────────────────
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}    GroundingDINO service started successfully!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Service status:"
echo "  GroundingDINO LB: http://localhost:$DINO_PORT  (PID: $DINO_LB_PID)"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "    Worker $i (GPU ${GPU_IDS_ARR[$i]}): http://localhost:${DINO_WORKER_PORTS[$i]}  (PID: ${DINO_WORKER_PIDS[$i]})"
done