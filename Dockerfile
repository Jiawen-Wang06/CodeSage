FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CODESAGE_DATA_DIR=/tmp/codesage \
    CODESAGE_BYOK_ONLY=true \
    PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=appuser:appuser .streamlit ./.streamlit
COPY --chown=appuser:appuser codesage ./codesage
COPY --chown=appuser:appuser code_sage.py codeSage.png ./

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/_stcore/health', timeout=4)"

CMD ["sh", "-c", "exec streamlit run code_sage.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true --server.fileWatcherType none"]
