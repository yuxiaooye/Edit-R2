#!/bin/bash


set -e

PYTHON_BIN="${PYTHON_BIN:-python3}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

MODEL_PATH="$1"
GPU_IDS_STR="${2:-0,1,2,3}"

IFS=',' read -ra GPU_IDS_ARR <<< "$GPU_IDS_STR"
NUM_GPUS=${#GPU_IDS_ARR[@]}

export CUDA_VISIBLE_DEVICES="$GPU_IDS_STR"

TP_SIZE=$NUM_GPUS

VLLM_PORT=8000
EDIVAL_PORT=12342

export VLLM_MODEL_PATH="$MODEL_PATH"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'


# ─────────────────────────────────────────────
# [1/3] Check dependencies
# ─────────────────────────────────────────────
echo -e "${YELLOW}[1/3] Checking dependencies...${NC}"

if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo -e "${RED}Error: $PYTHON_BIN not found. You can specify the python path via the PYTHON_BIN environment variable${NC}"
    exit 1
fi
if ! "$PYTHON_BIN" -c "import vllm" 2>/dev/null; then
    echo -e "${RED}Error: vllm is not installed. Please install it first: pip install vllm${NC}"
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: model path does not exist: $MODEL_PATH${NC}"
    exit 1
fi
if [ "$NUM_GPUS" -lt 1 ]; then
    echo -e "${RED}Error: GPU_IDs cannot be empty${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Dependency check passed${NC}"

# ─────────────────────────────────────────────
# [2/3] Clean up existing processes
# ─────────────────────────────────────────────
echo -e "${YELLOW}[2/3] Cleaning up existing processes...${NC}"
pid=$(lsof -ti:$VLLM_PORT 2>/dev/null || true)
if [ ! -z "$pid" ]; then
    echo "  Killing process occupying port $VLLM_PORT (PID: $pid)"
    kill -9 $pid 2>/dev/null || true
fi
sleep 2
echo -e "${GREEN}✓ Port cleaned up${NC}"

# ─────────────────────────────────────────────
# [3/3] Start the vLLM service
# ─────────────────────────────────────────────
echo -e "${YELLOW}[3/3] Starting vLLM service (TP=${TP_SIZE})...${NC}"
echo "  Model path: $MODEL_PATH"
echo "  Port:       $VLLM_PORT"
echo ""

LOG_FILE="vllm_server_${TIMESTAMP}.log"

nohup "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --port $VLLM_PORT \
    --trust-remote-code \
    --limit-mm-per-prompt "image=4" \
    > "$LOG_FILE" 2>&1 &


VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID"
echo ""
echo "  Waiting for vLLM to finish starting (duration depends on model size and machine performance)..."
echo -n "  ["
for i in {1..150}; do
    sleep 3
    if ! ps -p $VLLM_PID > /dev/null; then
        echo -e "${RED}]${NC}"
        echo -e "${RED}Error: vLLM process has exited. Check the log: tail -f $LOG_FILE${NC}"
        exit 1
    fi
    if curl -s "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
        echo -e "${GREEN}✓] Started successfully!${NC}"
        break
    fi
    echo -n "."
done

if ! curl -s "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
    echo -e "${RED}Error: vLLM service startup timed out. Check the log: tail -f $LOG_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ vLLM service is ready${NC}"
echo "  Log file: $LOG_FILE"
echo ""