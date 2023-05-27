import logging
from itertools import cycle
from typing import Iterable, Optional

import pandas as pd

from .base import Parametrizable, Serializable

logger = logging.getLogger(__name__)


class MostPopular(Parametrizable, Serializable):
    """
    A baseline recommendation model based on the most
    popular item in the train set.
    """

    def __init__(
        self,
        top_k: int = 100,
        movie_column: str = "movieId",
        user_column: str = "userId",
    ):
        """
        Initializes an instance of the MostPopular class.

        Args:
            top_k (int): The number of top items to consider for recommendation.
                         Defaults to 100.
            movie_column (str): The name of the column containing
                                movie IDs. Defaults to "movieId".
            user_column (str): The name of the column containing
            user IDs. Defaults to "userId".
        """
        self.top_k = top_k
        self.user_column = user_column
        self.movie_column = movie_column

    def fit(self, data: pd.DataFrame):
        """
        Fits the recommendation model to the provided data.
        This method computes the most popular movies based
        on the frequency of occurrence in the data.
        It keeps only self.top_k most popular movies.

        Args:
            data (pd.DataFrame): The input DataFrame containing user-movie interactions.
        """
        self.popularity = (
            data.groupby(self.movie_column)[self.user_column]
            .count()
            .sort_values(ascending=False)
            .head(self.top_k)
            .index
        )
        logger.debug(f"{MostPopular} is successfully fit.")

    def predict(self, n: int, seen_movies: Optional[Iterable[int]] = None):
        """
        Generates recommendations based on the most popular movies.
        The method excludes already viewed movies mentioned in the references.

        Args:
            seen_movies (list[int]): A list of already chosen item IDs.
            n (int): The number of recommendations to generate.

        Returns:
            list[int]: A list of item IDs representing the recommendations.
        """
        if seen_movies is None:
            seen_movies = []
        seen_movies = set(seen_movies)
        recs = []
        # here `cycle` is used to repeat movies
        # to pad predictions to required size
        for i, movie in enumerate(cycle(self.popularity)):
            is_first_cycle = i < len(self.popularity)

            # otherwise it leads to an infinite loop
            if is_first_cycle and movie in seen_movies:
                continue

            recs.append(int(movie))

            # exit condition
            if len(recs) == n:
                break

        return recs
