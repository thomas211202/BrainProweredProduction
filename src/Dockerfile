# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /src

# Copy the current directory contents into the container at /app
COPY ./requirements.txt /src/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /src/app

# Make port 80 available to the world outside this container
#EXPOSE 80
# Create a volume mount point
VOLUME /data

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app/main.py"]

#CMD ["gio", "open", "confusion_matrix.png"]
