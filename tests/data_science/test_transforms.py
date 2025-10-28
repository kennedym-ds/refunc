"""Tests for data science transformation utilities."""

import pandas as pd
import pytest

from refunc.data_science.transforms import (
    PipelineJoinConfig,
    apply_quick_preprocessing,
    create_basic_pipeline,
    join_dataframes_on_common_columns,
)
from refunc.exceptions import ValidationError


def test_join_dataframes_on_common_columns_inner() -> None:
    """Join multiple DataFrames on shared column names using inner join."""

    df_one = pd.DataFrame({"id": [1, 2], "feature_a": [10, 20]})
    df_two = pd.DataFrame({"id": [1, 2], "feature_b": [30, 40]})
    df_three = pd.DataFrame({"id": [1, 2], "feature_c": [50, 60]})

    result = join_dataframes_on_common_columns([df_one, df_two, df_three])

    expected = pd.DataFrame(
        {
            "id": [1, 2],
            "feature_a": [10, 20],
            "feature_b": [30, 40],
            "feature_c": [50, 60],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_join_dataframes_on_common_columns_outer() -> None:
    """Join DataFrames with an outer join and retain unmatched rows."""

    left = pd.DataFrame({"id": [1, 2], "score": [0.1, 0.2]})
    right = pd.DataFrame({"id": [2, 3], "value": [100, 200]})

    result = join_dataframes_on_common_columns([left, right], how="outer")

    assert set(result["id"].tolist()) == {1, 2, 3}
    assert pytest.approx(result.loc[result["id"] == 1, "score"].iloc[0]) == 0.1
    assert result.loc[result["id"] == 3, "score"].isna().all()


def test_join_dataframes_with_explicit_columns() -> None:
    """Join using explicitly provided columns and ensure validation occurs."""

    first = pd.DataFrame(
        {
            "id": [1, 2],
            "date": ["2024-01-01", "2024-01-02"],
            "value": [10.0, 20.0],
        }
    )
    second = pd.DataFrame(
        {
            "id": [1, 2],
            "date": ["2024-01-01", "2024-01-02"],
            "status": ["new", "closed"],
        }
    )

    result = join_dataframes_on_common_columns(
        [first, second],
        columns=["id", "date"],
        how="inner",
    )

    assert "status" in result.columns
    assert len(result) == 2

    third = pd.DataFrame({"id": [1, 2], "label": ["x", "y"]})
    with pytest.raises(ValidationError):
        join_dataframes_on_common_columns([first, third], columns=["id", "date"])


def test_join_dataframes_validate_and_indicator() -> None:
    """Validate join cardinality and request merge indicator column."""

    left = pd.DataFrame({"id": [1, 1], "feature": [10, 20]})
    right = pd.DataFrame({"id": [1], "metric": [30]})

    with pytest.raises(ValidationError):
        join_dataframes_on_common_columns([left, right], validate="one_to_one")

    outer_left = pd.DataFrame({"id": [1], "value": [5]})
    outer_right = pd.DataFrame({"id": [1, 2], "flag": [True, False]})

    indicator_result = join_dataframes_on_common_columns(
        [outer_left, outer_right],
        how="outer",
        indicator=True,
    )

    assert "_merge" in indicator_result.columns
    assert set(indicator_result["_merge"].unique()) == {"both", "right_only"}


def test_join_dataframes_without_common_columns_raises() -> None:
    """Raise ValidationError when inputs do not share column names."""

    df_one = pd.DataFrame({"left": [1, 2]})
    df_two = pd.DataFrame({"right": [3, 4]})

    with pytest.raises(ValidationError):
        join_dataframes_on_common_columns([df_one, df_two])


def test_pipeline_join_config_applies_before_steps() -> None:
    """Ensure pipeline join runs before configured transformation steps."""

    base = pd.DataFrame({"id": ["a", "b"], "score": [1.0, 2.0]})
    additional = pd.DataFrame({"id": ["a", "b"], "bonus": [10.0, 20.0]})

    pipeline = create_basic_pipeline(
        join_config=PipelineJoinConfig(sources=[additional])
    )

    result = pipeline.fit_transform(base)

    assert result.success is True
    assert result.final_data is not None
    assert "bonus" in result.final_data.columns
    assert any(
        step.transformation_name == "join_dataframes" for step in result.step_results
    )


def test_apply_quick_preprocessing_with_join_sources() -> None:
    """Quick preprocessing uses join shorthand and produces encoded columns."""

    primary = pd.DataFrame({"id": ["a", "b"], "score": [1.0, 2.0]})
    secondary = pd.DataFrame({"id": ["a", "b"], "status": ["new", "closed"]})

    transformed = apply_quick_preprocessing(
        primary,
        include_outlier_removal=False,
        join_sources=[secondary],
        join_columns=["id"],
        join_indicator=True,
    )

    status_columns = [column for column in transformed.columns if column.startswith("status_")]
    indicator_columns = [column for column in transformed.columns if column.startswith("_merge_")]

    assert status_columns, "Expected one-hot encoded status columns to be present"
    assert indicator_columns, "Expected indicator column encoding to be present"