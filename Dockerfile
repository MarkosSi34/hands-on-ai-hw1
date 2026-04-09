FROM ubuntu:24.04
 
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*
 
WORKDIR /app
 
RUN git clone -b main https://github.com/MarkosSi34/hands-on-ai-hw1.git .
RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir -r requirements-api.txt

EXPOSE 8080
 
CMD [".venv/bin/uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
 