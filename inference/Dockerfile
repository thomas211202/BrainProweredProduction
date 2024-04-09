# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /inference

# Copy the current directory contents into the container at /app
COPY ./requirements.txt /src/requirements.txt
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /inference/app

## Create a volume mount point
#VOLUME /data

# Run app.py when the container launches
CMD ["python", "app/main.py"]

