from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String
from datetime import timedelta

# Định nghĩa một entity. Trong trường hợp này, mỗi ngôi nhà là một entity.
house = Entity(name="house_id", value_type=ValueType.INT64, description="ID of the house")

# Định nghĩa nguồn dữ liệu (data source)
# Đây là file parquet chúng ta đã tạo
house_features_source = FileSource(
    path="data/house_features.parquet",
    event_timestamp_column="event_timestamp",
)

# Định nghĩa một Feature View. Đây là một nhóm các feature liên quan đến entity 'house'.
# Chúng ta sẽ định nghĩa một vài feature tiêu biểu.
# Bạn có thể thêm tất cả các cột còn lại vào đây.
house_features_view = FeatureView(
    name="house_features",
    entities=[house],
    ttl=timedelta(days=3650), # Time-to-live: dữ liệu cũ hơn 10 năm sẽ không được dùng
    schema=[
        Field(name="MSSubClass", dtype=Int64),
        Field(name="MSZoning", dtype=String),
        Field(name="LotFrontage", dtype=Float32),
        Field(name="LotArea", dtype=Int64),
        Field(name="Street", dtype=String),
        Field(name="OverallQual", dtype=Int64),
        Field(name="OverallCond", dtype=Int64),
        Field(name="YearBuilt", dtype=Int64),
        Field(name="YearRemodAdd", dtype=Int64),
        Field(name="TotalBsmtSF", dtype=Int64),
        Field(name="GrLivArea", dtype=Int64),
        Field(name="FullBath", dtype=Int64),
        Field(name="BedroomAbvGr", dtype=Int64),
        Field(name="TotRmsAbvGrd", dtype=Int64),
        Field(name="GarageCars", dtype=Int64),
        Field(name="GarageArea", dtype=Int64),
        Field(name="SalePrice", dtype=Int64), # Biến mục tiêu
    ],
    source=house_features_source,
    tags={"team": "sales_prediction"},
)