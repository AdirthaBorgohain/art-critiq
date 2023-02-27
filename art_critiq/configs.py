import os
from pathlib import Path
from typing import Dict, Union
from pydantic import BaseModel

root_dir = Path(__file__).parent.absolute().parent
data_dir = os.path.join(root_dir, "art_critiq/data")
review_models_dir = os.path.join(root_dir, "art_critiq/reviews/model_files")
print("root_dir: ", root_dir)

DEFAULT_CAPTION_MODELS = {
    "blip2": "Salesforce/blip2-opt-2.7b",
    "git": "microsoft/git-large-r-textcaps",
    "coca": "coca_ViT-L-14",
    "coca-pretrain": "mscoco_finetuned_laion2B-s13B-b90k"
}

REVIEW_MODEL_PATHS = {
    "KIND": os.path.join(review_models_dir, "kind_reviewer"),
    "CONSTRUCTIVE": os.path.join(review_models_dir, "constructive_reviewer"),
    "HARSH": os.path.join(review_models_dir, "harsh_reviewer"),
}

API_CONFIGS = {
    "host": "0.0.0.0",
    "port": 8080
}


class FastAPIClasses:
    class InferenceRequest(BaseModel):
        url: str = ""
        artist: str = ""
        title: str = ""
        type: str = ""
        technique: str = ""
        review_type: str = None

        class Config:
            schema_extra = {
                "example": {
                    "url": "https://openaccess-cdn.clevelandart.org/1915.534/1915.534_web.jpg",
                    "artist": "John Singleton Copley (American, 1738–1815)",
                    "title": "Nathaniel Hurd",
                    "type": "Painting",
                    "technique": "oil on canvas",
                    "review_type": "kind"
                }
            }

    class InferenceResponse(BaseModel):
        generated_review: Union[str, Dict]
        status_code: int

        class Config:
            schema_extra = {
                "example": {
                    "generated_review": "This beautiful hanging scroll painting, titled White-robed Kannon, "
                                        "is a stunning work of art. The painting is done in ink on silk and depicts a "
                                        "man in a traditional robe, surrounded by a vibrant array of colors and "
                                        "intricate details. The artist has expertly used light and shadow to create a "
                                        "sense of depth and movement in the painting. The details of the man's "
                                        "clothing and facial features are exquisitely rendered and the overall "
                                        "composition is balanced and pleasing to the eye. This painting is sure to be "
                                        "a beautiful addition to any home or office, and a great conversation starter.",
                    "status_code": 200
                }
            }


fastapi_classes = FastAPIClasses()
