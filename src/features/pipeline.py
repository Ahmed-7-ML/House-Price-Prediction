# Command -> uv run python -m src.features.pipeline
"""
    Scikit-learn feature preprocessing pipeline (scaler + optional poly).
"""

# ---> Imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

# ---> Define Numerical Features
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

# ---> Feature Preprocessor
def build_preprocessor(
    use_poly : bool = False,
    degree : int = 2
) -> ColumnTransformer:
    """
        Build a reusable preprocessing pipeline.
    """
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    
    if use_poly:
        numeric_steps.append(
            ("poly", PolynomialFeatures(degree=degree, include_bias=False))
        )
    
    numeric_transformer = Pipeline(steps=numeric_steps)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES)
        ],
        remainder="drop"
    )
    
    return preprocessor
