import torch
import requests
import open_clip
from PIL import Image
from art_critiq.captions.base import BaseCaptionGeneration
from art_critiq.configs import DEFAULT_CAPTION_MODELS


class CocaCaptionGeneration(BaseCaptionGeneration):
    def __init__(self, model_name_or_path: str = None, pretrained_set: str = None):
        """
        Args:
            model_name_or_path: Name of the coca model to load
        """
        super().__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.__transform, self.__model = self.__load_model(model_name_or_path or DEFAULT_CAPTION_MODELS['coca'],
                                                           pretrained_set or DEFAULT_CAPTION_MODELS['coca-pretrain'])
        self.__model.to(self.device)
        print("Successfully initialized caption generation model!")

    @staticmethod
    def __load_model(model_name: str, pretrained: str):
        coca_model, _, coca_transform = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained
        )
        return coca_transform, coca_model

    def generate_caption(self, url: str) -> str:
        image = Image.open(requests.get(url, stream=True).raw)
        im = self.__transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            generated = self.__model.generate(im, seq_len=25)
        return open_clip.decode(generated[0].detach()).split("<end_of_text>")[0].replace("<start_of_text>", "").replace(
            " .", ".")


if __name__ == '__main__':
    caption_generator = CocaCaptionGeneration()
    caption = caption_generator.generate_caption(
        url="https://d2jv9003bew7ag.cloudfront.net/uploads/Charles-Mayton-Abstract-expression-as-a-redundant-statement-2014-triptych-Gouache-and-oil-on-canvas-158-x-161-cm.jpg")
    print("generated_caption: ", caption)
