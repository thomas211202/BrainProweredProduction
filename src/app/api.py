from fastapi import FastAPI
from queue import Queue
from typing import List
from sklearn.externals import joblib
from classes.drone import (
    Drone
)

app = FastAPI()

@app.post("/process/")
async def process_data(data: List[List[float]]):
    # Convert the input list of lists to a NumPy array
    clf = joblib.load("./data/model.pkl")
    # TODO: add drone instance
    d = None
    drone = Drone(d)
    np_array = np.array(data)

    # Check if the input is a valid NumPy array
    if not isinstance(np_array, np.ndarray):
        raise HTTPException(status_code=400, detail="Input is not a valid NumPy array")

    drone.move(clf.predict(data))
    # Process the data (replace this with your own processing logic)
    # For demonstration, we simply return a success message
    return {"message": "Data processed successfully", "status_code": 200}

