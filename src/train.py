import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

def train_model(data_path):
    """
    Huấn luyện và đánh giá mô hình từ dữ liệu đã được xử lý.
    """
    # 1. Đọc dữ liệu đã xử lý
    df = pd.read_csv(data_path)

    # 2. Chia dữ liệu thành train và test
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    # 3. Tách features (X) và target (y)
    X_train = train_df.drop("SalePrice", axis=1)
    y_train = train_df["SalePrice"]
    X_test = test_df.drop("SalePrice", axis=1)
    y_test = test_df["SalePrice"]

    # Loại bỏ cột Id vì nó không phải là feature
    X_train = X_train.drop("Id", axis=1)
    X_test = X_test.drop("Id", axis=1)

    # 4. Định nghĩa các bước tiền xử lý bằng Pipeline
    # Xác định các cột số và cột hạng mục
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include='object').columns.tolist()

    # Pipeline cho các feature số
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    # Pipeline cho các feature hạng mục
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='none')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Kết hợp các pipeline bằng ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Giữ lại các cột không được xử lý (nếu có)
    )

    # 5. Tạo pipeline hoàn chỉnh: Preprocessing -> Model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # 6. Huấn luyện mô hình
    model_pipeline.fit(X_train, y_train)

    # 7. Đánh giá mô hình
    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Model evaluation:")
    print(f"  RMSE: {rmse}")
    print(f"  R2 Score: {r2}")

if __name__ == '__main__':
    train_model('data/processed/processed_data.csv')
