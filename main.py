from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
import uvicorn

app = FastAPI()
@app.get("/")
def cook():
    return Response(status_code=200)
class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: int
    max_power: int
    torque: str
    seats: int

with open('C:/Users/X/Desktop/linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)


def predict(item: Item):
    data = {
        'year': [item.year],
        'km_driven': [item.km_driven],
        'mileage': [item.mileage],
        'engine': [item.engine],
        'max_power': [item.max_power],
        'seats': [item.seats]
    }

    prediction = model.predict(pd.DataFrame(data))
    return prediction


@app.post("/predict_item", response_model=dict)
def predict_item(item: Item):
    return {"Предсказанная цена": predict(item)[0]}

if __name__ == '__main__':
    uvicorn.run('main:app', reload=True, port=7777)