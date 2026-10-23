# One Dockerfile, two images:
#   train: simulator + features + models (with PyTorch for the 1D CNN), writes artifacts/
#   app:   the Streamlit app, light (no PyTorch), reads artifacts/
#
#   docker build --target train -t treadcast-train .
#   docker build --target app   -t treadcast-app .
# or simply: docker compose run --rm pipeline && docker compose up app

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pyproject.toml .
COPY treadcast/ treadcast/

FROM base AS train
COPY requirements-train.txt .
RUN pip install -r requirements-train.txt
COPY tests/ tests/
CMD ["python", "-m", "treadcast", "--out", "/app/artifacts"]

FROM base AS app
RUN useradd --create-home --uid 1000 app
COPY .streamlit/ .streamlit/
COPY app.py .
COPY artifacts/ artifacts/
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
