import pandas as pd 
import numpy as np
import requests
import time
import joblib
import warnings


from evidently import Report
from evidently.presets import DataDriftPreset

warnings.filterwarnings('ignore')




# -----------------------
# تولید داده مصنوعی
# -----------------------
def generate_synthetic_data(original_data, num_samples=1000):
    synthetic_data = pd.DataFrame()


    for col in original_data.columns:
        if col not in ['anomaly_score', 'is_anomaly']:
            noise = np.random.normal(0, 0.1, num_samples)
            synthetic_data[col] = original_data[col].sample(num_samples, replace=True).reset_index(drop=True) + noise


    return synthetic_data




# -----------------------
# بارگذاری داده‌ها
# -----------------------
df_original = pd.read_csv('compressor_data_with_anomaly.csv', index_col=0)
df_original.index = pd.to_datetime(df_original.index)


# تولید داده مصنوعی
df_synthetic = generate_synthetic_data(df_original.drop(['anomaly_scores', 'is_anomaly'], axis=1))



# -----------------------
# بخش ۱: گزارش Evidently
# -----------------------
data_drift_report = Report([DataDriftPreset(method="psi")],include_tests="True")


my_eval = data_drift_report.run(
    reference_data=df_original.iloc[:1000],  # داده مرجع
    current_data=df_synthetic.iloc[:1000]    # داده مصنوعی
)


my_eval.save_json('data_drift_report.json')
print("✅ گزارش Evidently ذخیره شد: data_drift_report.json")





# -----------------------
# بخش ۳: ارسال داده‌ها به API
# -----------------------
API_URL = "http://127.0.0.1:8000/predict"  # مطمئن شو app.py در حال اجراست


print("🚀 شروع ارسال داده مصنوعی به API...")


for i in range(2000):  # 2000 رکورد تست
    sample = df_synthetic.iloc[i].tolist()  # یک ردیف
    try:
        resp = requests.post(API_URL, json={"sensor_data": sample})
        if resp.status_code == 200:
            print(f"[{i}] پاسخ API:", resp.json())
        else:
            print(f"[{i}] خطا:", resp.status_code, resp.text)
    except Exception as e:
        print(f"[{i}] ❌ مشکل در ارتباط با API:", e)


    time.sleep(1)  # هر ثانیه یک بار ارسال
