"""
    Pydantic schemas for request / response validation.
"""
from pydantic import BaseModel, Field

# ---> Define the Request/Input Format
class HouseFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group", ge=0)
    HouseAge: float = Field(..., description="Median house age", ge=0, le=100)
    AveRooms: float = Field(..., description="Average number of rooms", ge=0)
    AveBedrms: float = Field(..., description="Average number of bedrooms", ge=0)
    Population: float = Field(..., description="Block group population", ge=0)
    AveOccup: float = Field(..., description="Average household occupancy", ge=0)
    Latitude: float = Field(..., description="Block group latitude", ge=32, le=42)
    Longitude: float = Field(..., description="Block group longitude", ge=-125, le=-114)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "MedInc": 8.3252,
                    "HouseAge": 41.0,
                    "AveRooms": 6.9841,
                    "AveBedrms": 1.0238,
                    "Population": 322.0,
                    "AveOccup": 2.5556,
                    "Latitude": 37.88,
                    "Longitude": -122.23,
                }
            ]
        }
    }

# ---> Define the Response/Output Format
class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted median house value (×$100k)")
    currency_hint: str = "Value is in units of $100,000 (e.g. 2.5 ≈ $250,000)"
    model_version: str = "1.0.0"
