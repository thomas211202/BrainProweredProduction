# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /src

# Copy the current directory contents into the container at /app
COPY ./requirements.txt /src/requirements.txt
COPY ./config.yaml /src/config.yaml
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /src/app

# Create a volume mount point
VOLUME /data

# Run app.py when the container launches
CMD ["python", "app/main.py"]

