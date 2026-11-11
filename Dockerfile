FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip --version && \
    python -m pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY run_agent.py budget_tracker.py llm_client.py task_router.py ./
COPY workers/ workers/
COPY check_api_usage.py /app/check_api_usage.py

ENV BUDGET=100000
ENTRYPOINT ["python", "/app/run_agent.py"]