import pandas as pd
import pytest

from moviesurfer.metrics.classification import MeanAveragePrecisionTopK, PrecisionTopK


@pytest.fixture
def gt():
    return pd.DataFrame(
        data=[
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 4],
            [2, 5],
            [3, 6],
            [3, 7],
        ],
        columns=["user", "movie"],
    )


@pytest.fixture
def pred():
    return pd.DataFrame(
        data=[
            [1, 5, 1],
            [1, 4, 2],
            [1, 47, 3],
            [1, 2, 4],
            [1, 1, 5],
            [2, 1, 1],
            [2, 2, 2],
            [2, 3, 3],
            [2, 51, 4],
            [2, 47, 5],
            [3, 6, 1],
            [3, 52, 2],
            [3, 53, 3],
            [3, 54, 4],
            [3, 55, 5],
            [3, 56, 6],
        ],
        columns=["user", "movie", "rank"],
    )


def test_precision_fails_unknown_columns(pred, gt, k=1):
    precision = PrecisionTopK(k=k)
    with pytest.raises(Exception):
        precision(pred, gt)


@pytest.mark.parametrize(
    "k, expected",
    [
        [1, 1],
        [2, 10 / 12],
        [3, 8 / 12],
        [4, 7 / 12],
        [5, 8 / 15],
        [6, 8 / 18],
    ],
)
def test_precision(pred, gt, k, expected):
    precision_fn = PrecisionTopK(
        k=k, user_column="user", movie_column="movie", rank_column="rank"
    )
    metric_value = precision_fn(pred, gt)
    assert metric_value == expected


def test_map(pred, gt):
    map_fn = MeanAveragePrecisionTopK(
        k=5, user_column="user", movie_column="movie", rank_column="rank"
    )
    metric_value = map_fn(pred, gt)
    assert metric_value == (
        ((1 + 1 + 3 / 4 + 4 / 5) / 5 + (1 + 1 + 1) / 5 + (1) / 2) / 3
    )
