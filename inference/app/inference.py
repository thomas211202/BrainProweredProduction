import requests
import json
import numpy as np
import queue
import threading

# Define your API endpoint
API_URL = "http://localhost:8000/process/"

# Define a function to process data using the API
def process_data(data):
    try:
        # Send a POST request to the API endpoint with the data
        response = requests.post(API_URL, json=data)

        # Check if the request was successful
        if response.status_code == 200:
            print("Data processed successfully")
        else:
            print(f"Failed to process data: {response.json()}")
    except Exception as e:
        print(f"Error occurred: {e}")

# Define a function to continuously load data from the queue and process it
def worker():
    while True:
        try:
            # Get data from the queue
            data = data_queue.get()

            # Check if data is available
            if data is None:
                break

            # Process the data
            process_data(data)

            # Mark the task as done
            data_queue.task_done()
        except Exception as e:
            print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Create a queue to store data
    data_queue = queue.Queue()

    # Start worker threads
    num_threads = 4  # You can adjust the number of threads as needed
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=worker)
        thread.start()
        threads.append(thread)

    # Sample data to be processed
    sample_data = [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        # Add more sample data as needed
    ]

    # Enqueue sample data for processing
    for data in sample_data:
        data_queue.put(data)

    # Wait for all tasks to be processed
    data_queue.join()

    # Stop worker threads by adding None to the queue
    for _ in range(num_threads):
        data_queue.put(None)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
