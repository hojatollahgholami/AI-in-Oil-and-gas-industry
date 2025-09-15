import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.metrics import mean_squared_error
import joblib
 
df = pd.read_excel("CentrifugalCompressor2022.xlsx")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
print('مقادیر گمشده هر ستون:', df.isnull().sum())
df = df.interpolate(method='time')
print('اطلاعات کلی داده:',df.shape,',',df.describe())
#print(df.columns.tolist())
df.head(2)

plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap= 'coolwarm')
plt.title('نقشه همبستگی بین سنسورها')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

#standardation
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# model
model = IsolationForest(n_estimators=100,
			contamination=0.05, # فرض شده ۵درصد داده ها ناهنجار است
			random_state=42)
			
model.fit(scaled_data)

anomaly_scores = model.decision_function(scaled_data)
anomaly_pred = model.predict(scaled_data)

# add results to dataset
df['anomaly_scores'] = anomaly_scores
df['is_anomaly'] = anomaly_pred

# save model and scaler
joblib.dump(model, "anomaly_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# save data with anomaly results
df.to_csv('compressor_data_with_anomaly.csv')

#plot anomaly during time
plt.figure(figsize=(16, 8))
plt.plot(df.index,df['anomaly_scores'], label='anomaly score')
plt.scatter(df.index[df['is_anomaly'] == -1],
			df['anomaly_scores'][df['is_anomaly'] == -1],
			color='red', label='anomalies')
plt.title('نمره ناهنجاری در طول زمان')
plt.xlabel('تاریخ')
plt.ylabel('نمره ناهنجاری')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('anomaly_scores_over_time.png')
plt.close()
