FROM python:3.13.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MOTIF_FEEDBACK_HOST=0.0.0.0 \
    MOTIF_FEEDBACK_PORT=8000 \
    PATH=/opt/venv/bin:$PATH

RUN groupadd --gid 10001 motif \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin motif \
    && python -m venv /opt/venv

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY --chown=10001:10001 motif_feedback ./motif_feedback
RUN pip install --upgrade pip \
    && pip install --no-cache-dir .

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()" || exit 1

CMD ["motif-feedback"]
