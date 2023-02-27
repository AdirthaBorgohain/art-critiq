from abc import abstractmethod


class BaseCaptionGeneration:
    def __init__(self):
        pass

    @abstractmethod
    def generate_caption(self, url: str) -> str:
        """Generates a caption from the given url string"""
