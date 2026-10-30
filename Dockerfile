# Runs the triage tool in a container. The editor's MCP server stays on the host
# (it binds 127.0.0.1), so point TEGE_MCP_URL at host.docker.internal.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 1000 triage
WORKDIR /app

COPY pyproject.toml README.md ./
COPY triage ./triage
COPY eval ./eval
RUN pip install .

USER triage
VOLUME ["/app/reports"]

ENTRYPOINT ["enjin-triage"]
CMD ["--help"]
