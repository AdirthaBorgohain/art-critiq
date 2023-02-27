import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from art_critiq.configs import DEFAULT_CAPTION_MODELS


class GitCaptionGeneration:
    def __init__(self, model_name_or_path: str = None):
        """
        Args:
            model_name_or_path: Name of the model to load (from huggingface) or path to any local GIT Model
        """
        super().__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.__processor, self.__model = self.__load_model(model_name_or_path or DEFAULT_CAPTION_MODELS['git'])
        self.__model.to(self.device)
        print("Successfully initialized caption generation model!")

    @staticmethod
    def __load_model(model_name_or_path: str):
        """
        Loads the GIT model as specified
        Args:
            model_name_or_path: Name of the model to load (from huggingface) or path to any local GIT Model

        Returns:
            Image processor as well as the Causal LM model
        """
        git_processor = AutoProcessor.from_pretrained(model_name_or_path)
        git_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        return git_processor, git_model

    def generate_caption(self, url: str):
        """
        This function generates a caption by looking at the image of the artwork. The artwork is specified as an url,
        which is downloaded first and then used to generate a caption according to the contents of the artwork.
        Args:
            url: Valid and publicly accessible url for the artwork

        Returns:
            Generated caption from the image url
        """
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = self.__processor(images=image, return_tensors="pt").to(self.device)
        generated_ids = self.__model.generate(pixel_values=inputs.pixel_values, max_length=50)
        generated_caption = self.__processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_caption


if __name__ == '__main__':
    caption_generator = GitCaptionGeneration()
    caption = caption_generator.generate_caption(
        url="https://openaccess-cdn.clevelandart.org/1942.643/1942.643_web.jpg")
    print("generated_caption: ", caption)
