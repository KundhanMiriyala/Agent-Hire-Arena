FROM python:3.11-slim

# Hugging Face Spaces runs as a non-root user.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy source after dependencies.
COPY --chown=user . .

# HF Spaces expects port 7860.
EXPOSE 7860

# Run the FastAPI server.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
