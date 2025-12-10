# Use Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy source code and model
COPY ./src ./src
COPY ./models ./models
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
