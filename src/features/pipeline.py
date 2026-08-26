"""
Scikit-learn feature preprocessing pipeline (impute + scale + optional polynomial).
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

NUMERICAL_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "RoomsPerHousehold",
    "BedroomsPerRoom",
    "PopulationPerHousehold",
]


def build_preprocessor(
    use_poly: bool = False,
    degree: int = 2,
    feature_names: list[str] | None = None,
) -> ColumnTransformer:
    """Build a reusable numeric preprocessing pipeline."""
    columns = feature_names or NUMERICAL_FEATURES
    numeric_steps: list[tuple] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if use_poly:
        numeric_steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))

    numeric_transformer = Pipeline(steps=numeric_steps)
    return ColumnTransformer(
        transformers=[("num", numeric_transformer, columns)],
        remainder="drop",
    )
