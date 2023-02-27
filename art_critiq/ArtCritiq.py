"""
The ArtCritiq class combines both the caption generator and review generator together and is the primary POC (Point Of
Communication) with the API. Based on the user inputs, it first generates a caption for the artwork and then proceeds to
generate a review for the artwork.
"""
from typing import Union, Dict
from art_critiq.captions import Blip2CaptionGeneration, GitCaptionGeneration, CocaCaptionGeneration
from art_critiq.reviews import FlanT5ReviewGeneration


class ArtCritiq:
    def __init__(self, caption_model: str = 'blip2'):
        """
        Args:
            caption_model: Name of the caption model to use. Valid values are: "blip2, git, coca". Defaults to blip2.
        """
        self.__caption_models = {
            "blip2": Blip2CaptionGeneration,
            "git": GitCaptionGeneration,
            "coca": CocaCaptionGeneration
        }
        if caption_model not in self.__caption_models:
            raise ValueError(
                f"Invalid value for {caption_model} passed. Must be one of {list(self.__caption_models.keys())}")
        self.__caption_generator = self.__caption_models[caption_model]()
        self.__review_generator = FlanT5ReviewGeneration()
        self.valid_review_types = ['kind', 'harsh', 'constructive']

    def critque_art(self, url: str, artist: str, title: str, type: str, technique: str, review_type: str = None) -> \
            Union[str, Dict]:
        """
        This function takes in a bunch of parameters about the artwork as input and generates a review for the specified
        review type.

        Args:
            url: Valid and publicly accessible url for the artwork
            artist: Name of the artist
            title: Title of the artwork
            type: Type of artwork (eg. Painting, Sculpture, etc.)
            technique: Technique used in the artwork
            review_type: Type of review that is to be generated. Can be 'kind', 'constructive' or 'harsh'. If not
                         specified, it will generate reviews for all the types

        Returns:
            Generated review(s) according to user specified inputs. If multiple review types are generated, a dictionary
            is returned instead of a string.
        """
        if review_type and review_type not in self.valid_review_types:
            valid_categories_str = ", ".join(self.valid_review_types)
            raise ValueError(
                f"Invalid value for category ({review_type}) passed. Must be one of {valid_categories_str}"
            )
        generated_caption = self.__caption_generator.generate_caption(url=url)
        return self.__review_generator.generate_review(
            **{'artist': artist, 'title': title, 'type': type, 'technique': technique, 'caption': generated_caption,
               'review_type': review_type})


if __name__ == "__main__":
    critiq = ArtCritiq(caption_model="coca")
    review = critiq.critque_art(url="https://openaccess-cdn.clevelandart.org/1942.645/1942.645_web.jpg",
                                artist="Joshua Reynolds (British, 1723–1792)",
                                title="Portrait of the Ladies Amabel and Mary Jemima Yorke",
                                type="Painting", technique="Oil on canvas", review_type="kind")
    print("AI generated review for the artwork: ", review)
