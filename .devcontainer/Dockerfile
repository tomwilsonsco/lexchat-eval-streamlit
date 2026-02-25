ARG VARIANT=3.12-slim
FROM python:${VARIANT}

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

WORKDIR /app

# Copy and install requirements first to leverage build cache
COPY .devcontainer/requirements.txt .
RUN pip install --requirement requirements.txt --no-cache-dir

COPY . .

EXPOSE 8501

# ENTRYPOINT
ENTRYPOINT ["bash"]