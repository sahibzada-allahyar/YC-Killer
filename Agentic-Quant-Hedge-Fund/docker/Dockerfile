FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install -e .

# Copy source code
COPY src/ src/

# Set up environment
ENV PYTHONPATH=/app/src

CMD ["python", "-m", "agentic_quant.cli", "--help"]

