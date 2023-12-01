from fastapi import FastAPI
from .api.download import download_from_hf 

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
