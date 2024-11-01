# Start from the official Python 3.12 image based on Debian Bullseye
FROM python:3.12-bullseye

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file into the container
COPY . /app/
# COPY requirements.txt setup.py retrotide /app/

# Install Python dependencies from requirements.txt
RUN pip install -U pip
RUN pip install --no-cache-dir -r requirements.txt

# Install your package
RUN pip install .