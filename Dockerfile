# Use a specific, lightweight Python base image and set the platform for AMD64 compatibility.
FROM --platform=linux/amd64 python:3.9-slim-buster

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file and install dependencies.
# --no-cache-dir keeps the image size smaller.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project source code into the container.
COPY . . 

# Define the command to execute when the container starts.
# This will run the main script to process the PDFs.
CMD ["python", "main.py"]