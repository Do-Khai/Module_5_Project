import pandas as pd
from datetime import datetime

# Đọc dữ liệu đã xử lý
df = pd.read_csv('data/processed/processed_data.csv')

# Feast yêu cầu cột timestamp và cột entity
# Chúng ta sẽ dùng 'Id' làm entity 'house_id'
df.rename(columns={'Id': 'house_id'}, inplace=True)

# Tạo cột timestamp giả, vì dữ liệu này là tĩnh
df['event_timestamp'] = datetime.now()

# Lưu dưới dạng parquet
df.to_parquet('feature_repo/data/house_features.parquet')

print("Feast data prepared successfully.")
