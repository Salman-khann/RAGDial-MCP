# RAGDial MCP: reproducible image, served over Streamable HTTP on :8000/mcp
#
#   docker build -t ragdial-mcp .                       # core (offline hash embedder, BM25, RRF)
#   docker build -t ragdial-mcp:ml --build-arg EXTRAS=ml .   # + bge-small, bge-reranker-base, ChromaDB
#   docker run --rm -p 8000:8000 -v ragdial:/data ragdial-mcp
FROM python:3.11-slim

ARG EXTRAS=""
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    RAGDIAL_HOME=/data \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    PATH="/app/.venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project $( [ -n "$EXTRAS" ] && echo "--extra $EXTRAS" )

COPY src ./src
COPY sample_data ./sample_data
RUN uv sync --frozen --no-dev $( [ -n "$EXTRAS" ] && echo "--extra $EXTRAS" ) \
 && python -c "from ragdial_mcp.tokens import get_tokenizer; print(get_tokenizer().name)"

VOLUME ["/data"]
EXPOSE 8000
CMD ["ragdial-mcp", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]
