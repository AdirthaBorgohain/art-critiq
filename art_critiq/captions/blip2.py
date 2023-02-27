import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from art_critiq.captions.base import BaseCaptionGeneration
from art_critiq.configs import DEFAULT_CAPTION_MODELS


class Blip2CaptionGeneration(BaseCaptionGeneration):
    def __init__(self, model_name_or_path: str = None):
        """
        Args:
            model_name_or_path: Name of the model to load (from huggingface) or path to any local Blip2 Model
        """
        super().__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.torch_dtype = torch.float if self.device == 'cpu' else torch.float16
        self.__processor, self.__model = self.__load_model(model_name_or_path or DEFAULT_CAPTION_MODELS['blip2'],
                                                           dtype=self.torch_dtype)
        self.__model.to(self.device)
        print("Successfully initialized caption generation model!")

    @staticmethod
    def __load_model(model_name_or_path: str, dtype: torch.dtype) -> (Blip2Processor, Blip2ForConditionalGeneration):
        """
        Loads the Blip2 model as specified
        Args:
            model_name_or_path: Name of the model to load (from huggingface) or path to any local Blip2 Model

        Returns:
            Image processor as well as the conditional generation model
        """
        blip_processor = Blip2Processor.from_pretrained(model_name_or_path)
        blip_model = Blip2ForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=dtype)
        return blip_processor, blip_model

    def generate_caption(self, url: str) -> str:
        """
        This function generates a caption by looking at the image of the artwork. The artwork is specified as an url,
        which is downloaded first and then used to generate a caption according to the contents of the artwork.
        Args:
            url: Valid and publicly accessible url for the artwork

        Returns:
            Generated caption from the image url
        """
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = self.__processor(images=image, return_tensors='pt').to(self.device, self.torch_dtype)
        generated_ids = self.__model.generate(**inputs)
        generated_caption = self.__processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_caption


if __name__ == '__main__':
    caption_generator = Blip2CaptionGeneration()
    caption = caption_generator.generate_caption(
        url="https://openaccess-cdn.clevelandart.org/1942.643/1942.643_web.jpg")
    print("generated_caption: ", caption)
