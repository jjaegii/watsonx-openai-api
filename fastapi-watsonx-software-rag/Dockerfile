# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Install necessary Python packages
RUN pip install --no-cache-dir fastapi uvicorn requests tabulate

# Expose port 8000 for external access
EXPOSE 8000

# Define environment variables for runtime configuration
ENV BASE_URL=""
ENV PROJECT_ID=""
ENV USERNAME=""
ENV PASSWORD=""
ENV MILVUS_SEARCH_API_URL=""
ENV PYTHONUNBUFFERED=1

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
