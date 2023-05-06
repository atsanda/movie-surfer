from itertools import cycle, takewhile

import pandas as pd


class MostPopular:
    def __init__(
        self,
        top_k: int = 100,
        movie_column: str = "movieId",
        user_column: str = "userId",
        rating_column: str = "rating",
    ):
        self.top_k = top_k
        self.user_column = user_column
        self.rating_column = rating_column
        self.movie_column = movie_column

    def fit(self, data: pd.DataFrame):
        self.popularity = (
            data.groupby(self.movie_column)[self.user_column]
            .count()
            .sort_values(ascending=False)
            .head(self.top_k)
            .index
        )

    def predict(self, references: list[int], n: int):
        references = set(references)
        recs = []
        for movie in takewhile(lambda x: len(recs) < n, cycle(self.popularity)):
            if movie in references:
                continue
            recs.append(int(movie))

        return recs
