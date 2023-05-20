import typing as tp

import pandas as pd


class DataFramesTopKPreprocessor:
    """
    A class for common preprocessing of pandas dataframes for top-K metrics.
    """

    def __init__(
        self,
        user_column: str = "userId",
        movie_column: str = "movieId",
        rank_column: str = "rank",
    ):
        """
        Initializes a DataFramesTopKPreprocessor instance.

        Args:
            user_column str: The column name for the user identifier.
            movie_column str: The column name for the movie identifier.
            rank_column str: The column name for the rank of recommendations.
        """
        self.user_column = user_column
        self.movie_column = movie_column
        self.rank_column = rank_column

    def apply(self, pred: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
        """
        Applies preprocessing to the prediction and ground truth dataframes.
        It indexes dataframes along user and movie columns, then merges
        using left join to gt.

        Args:
            pred (pd.DataFrame): The dataframe containing the predictions.
            gt (pd.DataFrame): The dataframe containing the ground truth data.

        Returns:
            pd.DataFrame: The preprocessed dataframe containing the merged data.

        """
        gt_indexed = gt[[self.user_column, self.movie_column]].set_index(
            [self.user_column, self.movie_column]
        )
        pred_indexed = pred[
            [self.user_column, self.movie_column, self.rank_column]
        ].set_index([self.user_column, self.movie_column])
        merged = pd.merge(
            gt_indexed, pred_indexed, left_index=True, right_index=True, how="left"
        )
        return merged


class TopKBase:
    """
    A base class for top-K metrics.
    """

    def __init__(
        self,
        k: int,
        user_column: tp.Optional[str] = "userId",
        movie_column: tp.Optional[str] = "movieId",
        rank_column: tp.Optional[str] = "rank",
    ):
        """
        Initializes a TopKBase instance.

        Args:
            k (int): The value of K for the top-K metrics.
            user_column (str, optional): The column name for the user identifier.
                                            Defaults to "userId".
            movie_column (str, optional): The column name for the movie identifier.
                                            Defaults to "movieId".
            rank_column (str, optional): The column name for the rank of
                                            recommendations. Defaults to "rank".
        """
        self.k = k
        self.user_column = user_column
        self.movie_column = movie_column
        self.rank_column = rank_column
        self.preprocessor = DataFramesTopKPreprocessor(
            user_column, movie_column, rank_column
        )


class PrecisionTopK(TopKBase):
    def __call__(self, pred: pd.DataFrame, gt: pd.DataFrame) -> float:
        """
        Calculates the Precision@K metric.

        Args:
            pred (pd.DataFrame): The dataframe containing the predictions.
            gt (pd.DataFrame): The dataframe containing the ground truth data.

        Returns:
            float: The Precision@K metric value.
        """
        merged = self.preprocessor.apply(pred, gt)
        merged[f"hit@{self.k}"] = merged[self.rank_column] <= self.k
        merged[f"hit@{self.k}/{self.k}"] = merged[f"hit@{self.k}"] / self.k
        # group by user
        precision = (
            merged.groupby(level=self.user_column)[f"hit@{self.k}/{self.k}"]
            .sum()
            .mean()
        )
        return precision


class MeanAveragePrecisionTopK(TopKBase):
    def __call__(self, pred: pd.DataFrame, gt: pd.DataFrame) -> float:
        """
        Calculates the Mean Average Precision@K metric.

        Args:
            pred (pd.DataFrame): The dataframe containing the predictions.
            gt (pd.DataFrame): The dataframe containing the ground truth data.

        Returns:
            float: The Mean Average Precision@K metric value.
        """
        merged = self.preprocessor.apply(pred, gt)
        user_movie_count = gt.groupby(self.user_column)[self.movie_column].count()
        merged = merged.loc[merged[self.rank_column] <= self.k]
        merged = merged.sort_values(by=[self.user_column, self.rank_column])
        merged["cumulative_rank"] = (
            merged.groupby(level=self.user_column).cumcount() + 1
        )
        merged["cumulative_rank"] = merged["cumulative_rank"] / merged[self.rank_column]
        map = (
            merged["cumulative_rank"].groupby(level=self.user_column).sum()
            / user_movie_count
        ).mean()
        return map
