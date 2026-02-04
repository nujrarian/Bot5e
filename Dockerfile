# Bot5e – D&D 5e Assistant  (Ollama + Streamlit in one container)
#
# GPU: if you have an NVIDIA GPU + nvidia-container-toolkit, uncomment the
#      deploy.resources block in docker-compose.yaml.  This image runs fine
#      on CPU; llama3.1 8B Q4_K_M is usable without a GPU, just slower.

FROM python:3.13-slim

# ---------- Ollama --------------------------------------------------------
RUN curl -fsSL https://ollama.com/install.sh | sh

# ---------- uv (package manager) -----------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ---------- Python dependencies -------------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# ---------- Application code & config ------------------------------------
COPY *.py config.yaml eval_dataset.json ./

# ---------- Startup script ------------------------------------------------
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Streamlit UI  |  Ollama API  (Ollama port only needed if you want to call the API directly from the host)
EXPOSE 8501 11434                

# ---------------------------------------------------------------------------
# Mount points – see docker-compose.yaml for the defaults:
#   /app/SRD-OGL_V5.1.pdf          source PDF      (bind-mount, required)
#   /root/.ollama                   Ollama cache    (named volume)
#   /root/.cache/huggingface        HF model cache  (named volume)
# ---------------------------------------------------------------------------

ENTRYPOINT ["/start.sh"]
