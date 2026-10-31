from typing import Any, Mapping

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prepare_datasets(
    datasets: Mapping[str, pd.DataFrame],
    target: str,
    custom_maps: Mapping[str, Mapping[Any, Any]] | None = None,
    impute_categoricals: tuple[bool, str] = (False, "Missing"),
    boolean_columns: list[str] | None = None,
) -> tuple[
    dict[str, pd.DataFrame],
    list[str],
    list[str],
    list[str],
    dict[int, Any] | None,
    dict[Any, int] | None,
]:
    """Apply shared tabular preparation and return feature metadata.

    Mappings are applied consistently to every dataset, except the target.
    Boolean columns are kept as nullable integer values (0/1); other object
    columns are converted to pandas categorical dtype. Boolean inference is
    intentionally limited to real bool dtypes and explicit column names.
    """
    prepared_dataset = {name: subset.copy() for name, subset in datasets.items()}
    mappings = custom_maps or {}
    explicit_boolean = set(boolean_columns or [])

    if mappings:
        for name, subset in prepared_dataset.items():
            for column, mapping in mappings.items():
                if column in subset.columns and column != target:
                    subset[column] = subset[column].map(mapping)

    boolean_features = set(explicit_boolean)
    for subset in prepared_dataset.values():
        boolean_features.update(
            column
            for column in subset.columns
            if column != target and pd.api.types.is_bool_dtype(subset[column])
        )

    for name, subset in prepared_dataset.items():
        for column in boolean_features:
            if column in subset.columns:
                subset[column] = subset[column].astype("Int8")

        categorical_columns = [
            column
            for column in subset.select_dtypes(exclude="number").columns
            if column != target and column not in boolean_features
        ]
        if impute_categoricals[0]:
            print(f"ℹ️ Imputing missing values for categorical columns (subset: {name})")
            for col in categorical_columns:
                subset[col] = subset[col].fillna(impute_categoricals[1])
        subset[categorical_columns] = subset[categorical_columns].astype("category")

    train = prepared_dataset["train"]
    categorical_features = [
        column
        for column in train.select_dtypes(include=["object", "category"]).columns
        if column != target
    ]

    id2label: dict[int, Any] | None = None
    label2id: dict[Any, int] | None = None
    if target in train.select_dtypes(exclude="number").columns:
        encoder = LabelEncoder()
        train[target] = encoder.fit_transform(train[target])
        for name, subset in prepared_dataset.items():
            if name == "extra":
                subset[target] = encoder.transform(subset[target])

        id2label = dict(enumerate(encoder.classes_))
        label2id = {label: index for index, label in id2label.items()}

    numeric_features = [
        column
        for column in train.select_dtypes(include="number").columns
        if column != target
    ]
    boolean_features = [column for column in train.columns if column in boolean_features]
    categorical_features = [column for column in categorical_features if column not in boolean_features]

    return (
        prepared_dataset,
        numeric_features,
        categorical_features,
        boolean_features,
        id2label,
        label2id,
    )