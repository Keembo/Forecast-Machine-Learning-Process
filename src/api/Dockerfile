# Dockerfile for FastAPI backend
FROM python:3.8

# Set the working directory to /app
WORKDIR /app

# Copy the backend-specific requirements file into the container at /app
COPY requirements.txt /app/requirements.txt

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
