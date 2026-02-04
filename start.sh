#!/bin/bash
# start.sh – boot sequence for the Bot5e container
#   1. Start Ollama server (background)
#   2. Wait for the Ollama API to accept connections
#   3. Pull the configured LLM if not already cached (no-op on subsequent starts)
#   4. Exec Streamlit in the foreground so the container stays alive and
#      receives SIGTERM correctly.

set -e

# ------------------------------------------------------------------
# 1. Ollama
# ------------------------------------------------------------------
echo "[bot5e] Starting Ollama..."
ollama serve &

echo "[bot5e] Waiting for Ollama API..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
done
echo "[bot5e] Ollama ready."

# ------------------------------------------------------------------
# 2. Model pull (reads model name from config.yaml)
# ------------------------------------------------------------------
MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('/app/config.yaml'))['llm']['model'])")
echo "[bot5e] Ensuring model: ${MODEL}"
ollama pull "${MODEL}"

# ------------------------------------------------------------------
# 3. Streamlit  (exec replaces this shell → PID 1 = streamlit)
# ------------------------------------------------------------------
echo "[bot5e] Launching Streamlit..."
exec streamlit run /app/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true
