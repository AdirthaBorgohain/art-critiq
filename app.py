"""
FastAPI script to run the art_critiq module. Currently has one working endpoint: /generate_review. Example of how to
make api calls through python is given in sample_api_call.py script.
"""
import uvicorn
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware

from art_critiq.ArtCritiq import ArtCritiq
from art_critiq.configs import API_CONFIGS, fastapi_classes

app = FastAPI()
origins = ['*']

critiq = ArtCritiq(caption_model="blip2")  # Valid values are "blip2, git, coca"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health_check")
def health_check(response: Response):
    pass


@app.post("/generate_review", response_model=fastapi_classes.InferenceResponse, response_model_exclude_none=True)
def generate_review(response: Response, request: fastapi_classes.InferenceRequest):
    try:
        generated_review = critiq.critque_art(url=request.url, artist=request.artist, title=request.title,
                                              type=request.type, technique=request.technique,
                                              review_type=request.review_type)
        status_code = 200
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        print("Error occurred: ", e)
        generated_review = ""
        status_code = 400
    return {"generated_review": generated_review, "status_code": status_code}


if __name__ == "__main__":
    uvicorn.run(app, host=API_CONFIGS["host"], port=API_CONFIGS["port"])
