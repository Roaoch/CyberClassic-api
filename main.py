import warnings

from src.cyberclaasic import CyberClassic
from fastapi import FastAPI

warnings.simplefilter("ignore", UserWarning)

app = FastAPI()

text_generator = CyberClassic(
    max_length=60,
    startings_path='./startings.csv'
)

@app.get("/")
def generete():
    return {"text": str(text_generator.generate())}

@app.get('/answer')
def answer(promt: str):
    return {"text": str(text_generator.answer(f'{promt}:\n'))}