from abc import abstractmethod


class BaseReviewGeneration:
    def __init__(self):
        pass

    @abstractmethod
    def generate_review(self, **kwargs) -> str:
        """Generates a review from the given keyword arguments"""
