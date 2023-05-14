import pandas as pd
import pytest

from moviesurfer.models import MostPopular


@pytest.fixture
def simple_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        data=[
            [1, 1],
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 2],
            [3, 4],
            [3, 1],
        ],
        columns=["userId", "movieId"],
    )


@pytest.fixture
def trained_most_popular_model(simple_train_df: pd.DataFrame):
    model = MostPopular()
    model.fit(simple_train_df)
    return model


def test_predicts_most_popular(trained_most_popular_model: MostPopular):
    recs = trained_most_popular_model.predict(n=1)
    assert len(recs) == 1
    assert recs[0] == 1


def test_excludes_already_viewed(trained_most_popular_model: MostPopular):
    recs = trained_most_popular_model.predict(references=[1], n=1)
    assert len(recs) == 1
    assert recs[0] == 2
