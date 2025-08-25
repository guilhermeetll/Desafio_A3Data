# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the frontend directory contents into the container
COPY . /app

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit when the container launches
CMD ["streamlit", "run", "frontend_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
