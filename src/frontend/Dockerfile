# Dockerfile for Streamlit frontend
FROM python:3.8

# Set the working directory to /app
WORKDIR /app

# Copy the frontend-specific requirements file into the container at /app
COPY requirements.txt /app/requirements.txt

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.headless=true", "--server.address=0.0.0.0"]
