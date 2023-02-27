import re
import torch
from typing import Union, Dict
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from art_critiq.reviews.base import BaseReviewGeneration
from art_critiq.configs import REVIEW_MODEL_PATHS


class FlanT5Model:
    def __init__(self, model_name_or_path: str):
        """
        Args:
            model_name_or_path: Name of the model to load (from huggingface) or path to any local FlanT5 Model
        """
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.__tokenizer = T5TokenizerFast.from_pretrained('google/flan-t5-base')  # Load tokenizer of FLAN-t5-base
        self.__model = self.__load_model(model_name_or_path)

    def __load_model(self, model_name_or_path: str):
        """
        Loads the T5 model as specified
        Args:
            model_name_or_path: Name of the model to load (from huggingface) or path to any local FlanT5 Model

        Returns:
            Loaded T5 model after putting in on the relevant device (CPU/GPU)
        """
        t5_model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        t5_model.to(self.device)
        return t5_model

    def generate_text(self, text: str):
        """
        Generates text from the loaded FlanT5 model
        Args:
            text: Text which is used as a conditional text for the model to generate the rest of the text

        Returns:
            Generated text based on the conditional input text
        """
        tokenized_outputs = self.__tokenizer(text, return_tensors='pt').to(self.device)
        model_output = self.__model.generate(**tokenized_outputs, do_sample=True, temperature=0.7, top_p=0.90,
                                             no_repeat_ngram_size=2, max_length=500)
        return self.__tokenizer.decode(model_output[0], skip_special_tokens=True)


class FlanT5ReviewGeneration(BaseReviewGeneration):
    def __init__(self):
        super().__init__()
        self.__ARTIST_PATTERN = re.compile(r'^\s*([^(\n]*)')
        self.__reviewer_models = {
            'kind': FlanT5Model(REVIEW_MODEL_PATHS['KIND']),
            'constructive': FlanT5Model(REVIEW_MODEL_PATHS['CONSTRUCTIVE']),
            'harsh': FlanT5Model(REVIEW_MODEL_PATHS['HARSH'])
        }
        print("Successfully initialized all review generation models!")

    def __create_description(self, **kwargs) -> str:
        """
        This function creates a description of the artwork based on the details given as input. It's basically formatted
        as a narrative which is similar to how the description was formatting during fine-tuning of the model.
        Args:
            **kwargs: Keyword arguments corresponding to the details of the artwork initially given as input.

        Returns:
            Created description which can be used as input to the FlanT5 model.
        """
        artist = kwargs.get('artist')
        if artist:
            match = self.__ARTIST_PATTERN.match(artist)
            artist = match.group(1).strip() if match else artist.strip()
        else:
            artist = "Unidentified Artist"

        description = (f'''The title of the artwork is "{kwargs.get('title', 'Unidentified')}". It is '''
                       f'''created by {artist} using the technique of {kwargs.get('technique', 'Unidentified')}. The '''
                       f'''artwork can be described as follows: "{kwargs.get('caption', '').strip()}"''')
        return description

    def generate_review(self, **kwargs) -> Union[str, Dict]:
        """
        This function takes in keyword arguments corresponding to the details of an artwork and generates a review from
        the description of the artwork using a fine-tuned T5 text generation model.
        Args:
            **kwargs: Keyword arguments corresponding to the details of the artwork initially given as input.

        Returns:
            Generated review from the description (created by the __create_description function)
        """
        description = self.__create_description(**kwargs)
        review_type = kwargs.get('review_type')
        if review_type:
            return self.__reviewer_models[review_type].generate_text(description)
        else:
            return {review_type: model.generate_text(description) for review_type, model in
                    self.__reviewer_models.items()}


if __name__ == '__main__':
    review_generator = FlanT5ReviewGeneration()
    generated_reviews = review_generator.generate_review(
        title="Wanderer above the Sea of Fog", artist="Caspar David Friedrich",
        technique="oil on canvas",
        caption="a man standing on top of a mountain looking out over the fog.", review_type=None)
    print("generated_reviews: ", generated_reviews)
