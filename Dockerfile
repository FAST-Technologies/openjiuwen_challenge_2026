FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run_agent.py budget_tracker.py llm_client.py task_router.py ./
COPY workers/ workers/

ENV BUDGET=100000
ENTRYPOINT ["python", "/app/run_agent.py"]