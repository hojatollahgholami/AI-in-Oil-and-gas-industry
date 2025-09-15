#app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List

app = FastAPI(title='compressor health monitoring API')

model = joblib.load('anomaly_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

class SensorData(BaseModel):
	sensor_data: List[float]   #مقدار سنسورها (۲۵سنسور) 

@app.post('/predict')
async def predict_anomaly(data: SensorData):
	try:
		# تبدیل داده ورودی به آرایه
		input_data = np.array(data.sensor_data).reshape(1,-1)
		
		# استانداردسازی
		scaled_input = scaler.transform(input_data)
		
		# \یش بینی ناهنجاری
		anomaly_score = model.decision_function(scaled_input)[0]
		is_anomaly = model.predict(scaled_input)[0]
		
		return {
				'anomaly_score': float(anomaly_score),
				'is_anomaly': int(is_anomaly),
				'anomaly_status': 'Anomlay' if is_anomaly == -1 else 'Normal'
				}
	except Exception as e:
		return {'error': str(e)}

@app.get('/health')
async def health_check():
	return {'status': 'healthy'}


# استفاده از کد زیر برای اجرای برنامه در \نجره دیگری
# uvicorn ex39_app:app --host 127.0.0.1 --port 8000
# س\س از آدرس های زیر در مرورگر می توان استفاده کرد
# http://127.0.0.1:8000/health
# http://127.0.0.1:8000/docs
