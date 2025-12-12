# Stage 1: Base (Now uses the Pre-Built Image)
# We assume 'social-media-backend-base:latest' fails if not built, 
# BUT for local dev we can just use it.
# Ideally, we push this base to DockerHub, but for local/minikube:
FROM python:3.9-slim AS base
WORKDIR /app
# Install system dependencies (Required for LightGBM/Scikit-learn)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Install dependencies directly (Fast now!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Test
FROM base AS test
COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt
COPY . .
# Runs tests instantly
RUN python -m pytest test_main.py

# Stage 3: Production
FROM base AS production
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
