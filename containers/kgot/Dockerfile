FROM spcleth/kgot:latest

WORKDIR /home/knowledge_graph_of_thoughts

# Adding minimum files to build dependencies
ADD pyproject.toml .
RUN mkdir -p kgot
ADD kgot/__init__.py kgot/__init__.py
ADD kgot/__main__.py kgot/__main__.py
ADD LICENSE .
ADD README.md .

# Required for building packages such as onnx from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    protobuf-compiler \
    libprotobuf-dev \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnss3 \
    libpango-1.0-0 \
    libx11-6 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Install kgot dependencies
RUN python3 -m venv venv && . venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install "onnx==1.22.0" "ml_dtypes>=0.5.4" && \
    pip install -e . && \
    playwright install

RUN . venv/bin/activate && kaleido_get_chrome

# langchain-community may not know this dated model name yet. Add the same
# pricing entries as gpt-5-mini so get_openai_callback can compute costs.
RUN venv/bin/python -c "from pathlib import Path; import langchain_community.callbacks.openai_info as oi; path = Path(oi.__file__); text = path.read_text(); marker = '    # GPT-5-mini output\n'; model = '\"gpt-5.4-mini-2026-03-17\"'; insert = '    # GPT-5.4-mini input\n    \"gpt-5.4-mini-2026-03-17\": 0.00075,\n    \"gpt-5.4-mini-2026-03-17-cached\": 0.000075,\n    # GPT-5.4-mini output\n    \"gpt-5.4-mini-2026-03-17-completion\": 0.0045,\n'; text.index(marker) if model not in text else None; path.write_text(text if model in text else text.replace(marker, insert + marker))"

# The copy process is done later, so that all Python dependencies are installed before copying the code and thus cached if code changes.
# Copy KGoT code to the container
COPY kgot kgot
COPY benchmarks benchmarks
ADD run_multiple_* .
RUN chmod +x run_multiple_*

ENTRYPOINT [ "/bin/bash", "-c"]
