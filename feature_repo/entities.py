from feast import Entity, ValueType

housing_entity = Entity(
    name = "entity_id",
    value_type=ValueType.INT64,
    description="Unique housing block identifier",
)