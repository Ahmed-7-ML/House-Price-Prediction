from datetime import timedelta
from feast import FeatureView, Field, FileSource
from feast.types import Float32, Int64
from entities import housing_entity

# Offline sources
housing_features_source = FileSource(
    path="data/housing_features.parquet",
    timestamp_field="event_timestamp",
)

housing_target_source = FileSource(
    path="data/housing_target.parquet",
    timestamp_field="event_timestamp",
)

housing_features_fv = FeatureView(
    name="housing_features",
    entities=[housing_entity],
    ttl=timedelta(days=365 * 10),
    schema=[
        Field(name="MedInc", dtype=Float32),
        Field(name="HouseAge", dtype=Float32),
        Field(name="AveRooms", dtype=Float32),
        Field(name="AveBedrms", dtype=Float32),
        Field(name="Population", dtype=Float32),
        Field(name="AveOccup", dtype=Float32),
        Field(name="Latitude", dtype=Float32),
        Field(name="Longitude", dtype=Float32),
        Field(name="RoomsPerHousehold", dtype=Float32),
        Field(name="BedroomsPerRoom", dtype=Float32),
        Field(name="PopulationPerHousehold", dtype=Float32),
    ],
    source=housing_features_source,
    online=True,
)

housing_target_fv = FeatureView(
    name="housing_target",
    entities=[housing_entity],
    ttl=timedelta(days=365 * 10),
    schema=[
        Field(name="MedHouseVal", dtype=Float32),
    ],
    source=housing_target_source,
    online=False,
)
