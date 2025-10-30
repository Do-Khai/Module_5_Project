import pandas as pd
import numpy as np
import argparse
import os

def process_data(raw_data_path, processed_data_path):
    """
    Đọc dữ liệu thô, xử lý giá trị thiếu và lưu lại.
    """
    df = pd.read_csv(raw_data_path)

    # Loại bỏ các cột không cần thiết hoặc có quá nhiều giá trị thiếu
    df = df.drop(["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1)

    # Xử lý giá trị thiếu đơn giản (dựa trên notebook của bạn)
    # Các cột số: điền bằng giá trị trung bình
    num_cols_with_na = df.select_dtypes(include=np.number).columns[df.select_dtypes(include=np.number).isnull().any()]
    for col in num_cols_with_na:
        df[col].fillna(df[col].mean(), inplace=True)

    # Các cột object: điền bằng 'None' hoặc giá trị mode
    obj_cols_with_na = df.select_dtypes(include='object').columns[df.select_dtypes(include='object').isnull().any()]
    for col in obj_cols_with_na:
        # Các cột này giá trị NA có ý nghĩa là "không có"
        if col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
            df[col].fillna('None', inplace=True)
        else: # Các cột khác điền bằng mode
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Tạo thư mục output nếu chưa có
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

    # Lưu dữ liệu đã xử lý
    df.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_path', help='Path to the raw CSV file')
    parser.add_argument('processed_path', help='Path to save the processed CSV file')
    args = parser.parse_args()
    process_data(args.raw_path, args.processed_path)
