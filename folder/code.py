import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import jdatetime
from jdatetime import date as jdate
from persiantools.jdatetime import JalaliDate
from datetime import date
import warnings
warnings.filterwarnings('ignore')
import arabic_reshaper
from bidi.algorithm import get_display
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from io import BytesIO
import base64
#import pulp
import math
from collections import defaultdict

# تابع کمکی برای نمایش صحیح متون فارسی
def fa(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# تنظیمات فارسی‌نویسی
plt.rcParams['font.size'] = 16
plt.rcParams["font.family"] = 'Sakkal Majalla' #'Adobe Arabic' 'Microsoft Uighur' 
plt.rcParams["axes.unicode_minus"] = False

file_path = 'merged_data.xlsx'
df = pd.read_excel(file_path)
print("✅ فایل Excel بارگذاری شد")
            
# بررسی ساختار داده‌ها
print("🔍 بررسی ساختار داده‌ها...")
print(f"ستون‌های موجود: {list(df.columns)}")

"""ایجاد ستون‌های ضروری اگر وجود ندارند"""
# مصرف حیاتی - استفاده از داده‌های تاریخی
if 'critical_demand' not in df.columns:
        critical_cols = ['9109 A (MW)', '9109 B (MW)', 'S / S 1 A (MW)', 'S / S 1 B (MW)',
                           'S / S 2 B (MW)', 'S / S 2A (MW)', 'S / S 3 B (MW)', 'S / S 4 A (MW)',
                           'S / S 4 B (MW)', 'S / S 5 A (MW)', 'S / S 5 B (MW)', 'S / S 6 A (MW)',
                           'S / S 6 B (MW)', 'S / S 7 A (MW)', 'S / S 7 B (MW)']
            
        df['critical_demand'] = 0
        for col in critical_cols:
            if col in df.columns:
                    df['critical_demand'] += pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        if 'kangan  sattalite(kw)' in df.columns:
            df['critical_demand'] += pd.to_numeric(df['kangan  sattalite(kw)'], errors='coerce').fillna(0) / 1000

# مصرف مسکونی
if 'living_demand' not in df.columns:
        living_cols = ['LIVING A (MW)', 'LIVING B (MW)']
        df['living_demand'] = 0
        for col in living_cols:
            if col in df.columns:
                df['living_demand'] += pd.to_numeric(df[col], errors='coerce').fillna(0) 

# پردازش تاریخ
def safe_shamsi_to_gregorian(date_str):
        """تبدیل ایمن تاریخ شمسی به میلادی"""
        try:
            if pd.isna(date_str):
                return pd.NaT
                
            date_str = str(date_str).strip()
            date_str = ''.join(c for c in date_str if c.isdigit() or c in ['-', '/'])
            
            parts = date_str.replace('/', '-').split('-')
            if len(parts) == 3:
                year, month, day = parts
                year = int(year) if year.isdigit() else 1402
                month = int(month) if month.isdigit() else 1
                day = int(day) if day.isdigit() else 1
                
                year = max(1300, min(year, 1500))
                month = max(1, min(month, 12))
                day = max(1, min(day, 31))
                
                return jdatetime.date(year, month, day).togregorian()
                
        except Exception as e:
            print(f"⚠️ خطا در تبدیل تاریخ {date_str}: {e}")
            
        return pd.NaT
    
if 'time_gregorian' in df.columns:
                df['date'] = pd.to_datetime(df['time_gregorian'], errors='coerce')
                print("✅ از تاریخ میلادی موجود استفاده شد")
elif 'time_shamsi' in df.columns:
                df['date'] = df['time_shamsi'].apply(safe_shamsi_to_gregorian)
                print("✅ تاریخ‌های شمسی تبدیل شدند")
else:
                df['date'] = pd.date_range('2023-01-01', periods=len(df), freq='D')
                print("⚠️ تاریخ مصنوعی ایجاد شد")
            
initial_len = len(df)
df = df.dropna(subset=['date']).copy()
final_len = len(df)
print(f"📅 ردیف‌های حذف شده: {initial_len - final_len}")            
df.head()

"""پردازش داده‌های عددی"""
numeric_columns = df.select_dtypes(include=[np.number]).columns
        
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
plants = ['TG B', 'TG C', 'TG D', 'TG 5', 'TG 6', 'TG 7']

for plant in plants:
    if plant in df.columns:
        if df[plant].sum() == 0:
            print(f"⚠️ داده‌های {plant} صفر هستند - ایجاد داده‌های نمونه")
df.head()

"""پردازش ستون active_date براساس تاریخ تقویمی (پنجشنبه و جمعه)"""
if 'date' in df.columns:
            print("📅 پردازش ستون active_date براساس تاریخ تقویمی...")

            # اطمینان از اینکه ستون date نوع datetime است
            if not np.issubdtype(df['date'].dtype, np.datetime64):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # استخراج روز هفته: دوشنبه=0 ... یکشنبه=6
            df['weekday'] = df['date'].dt.weekday  

            # تعریف پنجشنبه و جمعه
            df['is_thursday'] = (df['weekday'] == 3).astype(int)  # پنجشنبه
            df['is_holiday'] = (df['weekday'] == 4).astype(int)   # جمعه

            # روز کاری: غیر از پنجشنبه و جمعه
            df['is_working_day'] = ((df['is_thursday'] == 0) & (df['is_holiday'] == 0)).astype(int)

            print(f"   روزهای کاری: {df['is_working_day'].sum()}")
            print(f"   پنجشنبه‌ها: {df['is_thursday'].sum()}")
            print(f"   جمعه‌ها: {df['is_holiday'].sum()}")


# تبدیل میلادی به شمسی
df['jalali_date'] = df['date'].apply(lambda x: JalaliDate(x))
df['jalali_year'] = df['jalali_date'].apply(lambda x: x.year)
df['jalali_month'] = df['jalali_date'].apply(lambda x: x.month)
df['jalali_day'] = df['jalali_date'].apply(lambda x: x.day)

#پیش بینی نیاز مسکونی با رگرسیون جنگل تصادفی
days_ahead = 365

"""پیش‌بینی مصرف با مدل ماشین لرنینگ"""
print("پیش‌بینی مصرف مسکونی با مدل ML...")
        
# آماده‌سازی داده‌ها
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['date'].dt.dayofweek.isin([4, 5])  # جمعه و پنجشنبه
        
# ویژگی‌ها
features = ['month', 'day_of_year', 'day_of_week', 'is_weekend']
if 'mean_temperature_2m' in df.columns:
    features.append('mean_temperature_2m')
    
if 'mean_relativehumidity_2m' in df.columns:
    features.append('mean_relativehumidity_2m')
    
X = df[features]
y = df['living_demand'] #جهت برآورد بیشتر با توجه به هدف مسئله
        
# آموزش مدل
model_residential_rf = RandomForestRegressor(n_estimators=400, random_state=42)
model_residential_rf.fit(X, y)
        
# پیش‌بینی روی داده آزمون
y_pred_test = model_residential_rf.predict(X)
        
# محاسبه معیارهای خطا
mae = mean_absolute_error(y, y_pred_test)
rmse = np.sqrt(mean_squared_error(y, y_pred_test))
r2 = r2_score(y, y_pred_test)
        
print(f"📈 ارزیابی مدل روی داده آزمون:")
print(f"   MAE: {mae:.2f}")
print(f"   RMSE: {rmse:.2f}")
print(f"   R²: {r2:.4f}")

"""محاسبه ظرفیت‌های قابل اطمینان"""
# محاسبه ظرفیت‌ها
capacities = {}
maxs = {}

for plant in plants:
    if plant in df.columns:
        plant_data = df[plant]
        capacities[plant] = np.percentile(plant_data, 97)
        maxs[plant] = max(plant_data)  # حالا کار می‌کنه چون تابع max بازتعریف نشده

# آماده‌سازی داده‌ها برای نمودار
plants = list(capacities.keys())
values = list(capacities.values())
max_values = list(maxs.values())  # به‌جای استفاده از اسم max
print(values,max_values)

plant_capacities = capacities
reliability = {}
for plant in plants:
    if plant in df.columns:
                plant_data = df[plant]
                capacity = plant_capacities.get(plant, 40)
                reliability[plant] = {
                    'stability_index': (plant_data > capacity * 0.5).mean(),
                    'availability_ratio': (plant_data > 0).mean()
                }
    else:
                reliability[plant] = {'stability_index': 0.0, 'availability_ratio': 0.0}

# رسم نمودار
plants = list(reliability.keys())
stability_indices = [reliability[p]['stability_index'] for p in plants]
availability_ratios = [reliability[p]['availability_ratio'] for p in plants]

cutoff_threshold = 5

# تعداد کل بازه‌های 10 روزه در سال شمسی = 12 ماه * 3 بازه در هر ماه = 36
num_periods = 12 * 3

# دیکشنری برای ذخیره ریسک هر بازه در همه سال‌ها
risks_per_period_all_years = {i: [] for i in range(num_periods)}

years = df['jalali_year'].unique()

for year in years:
    df_year = df[df['jalali_year'] == year]
    tavanir_data = df_year['Tavanir'].values
    months = df_year['jalali_month'].values
    days = df_year['jalali_day'].values
    
    for i in range(num_periods):
        month_idx = i // 3 + 1   # ماه شمسی (1 تا 12)
        day_block = i % 3        # بلاک روز: 0 (1-10), 1 (11-20), 2 (21-end)
        
        if day_block == 0:
            start_day, end_day = 1, 10
        elif day_block == 1:
            start_day, end_day = 11, 20
        else:
            start_day, end_day = 21, 31  # 31 به عنوان حداکثر روز هر ماه در نظر گرفته شده
        
        mask = (months == month_idx) & (days >= start_day) & (days <= end_day)
        period_data = tavanir_data[mask]
        
        if len(period_data) > 0:
            risk = (period_data <= cutoff_threshold).mean()
            risks_per_period_all_years[i].append(risk)
        else:
            pass  # اگر داده نداشتیم چیزی اضافه نکن
    
# میانگین ریسک هر بازه
avg_risks = []
for i in range(num_periods):
    if len(risks_per_period_all_years[i]) > 0:
        avg_risks.append(np.mean(risks_per_period_all_years[i]))
    else:
        avg_risks.append(0)

# مکان ماه‌ها روی محور افقی: هر 3 بازه = 1 ماه، پس وسط هر 3 بازه رو برای لیبل ماه می‌ذاریم
month_ticks = [i*3  for i in range(len(months_shamsi))]  # بازه‌ها از 0 شروع میشه، پس مرکز بازه‌های 3تایی

def gregorian_to_jalali(date):
    return jdatetime.date.fromgregorian(date=date).strftime('%Y-%m-%d')

df['time_gregorian'] = pd.to_datetime(df['time_gregorian'])
df['jalali_date'] = df['time_gregorian'].apply(gregorian_to_jalali)

maintenance_levels = [
    {'interval': 16000, 'level': 'درجه 5', 'color': 'darkred'},
    {'interval': 8000, 'level': 'درجه 4', 'color': 'yellow'},
    {'interval': 4000, 'level': 'درجه 3', 'color': 'blue'},
    {'interval': 2000, 'level': 'درجه 2', 'color': 'green'},
    {'interval': 500, 'level': 'درجه 1', 'color': 'red'},
]

for plant in plants:
    if plant in df.columns:
        df[f'{plant}_daily_hours'] = df[plant].apply(lambda x: 24 if x > 0 else 0)
        df[f'{plant}_cumulative_hours'] = df[f'{plant}_daily_hours'].cumsum()

        plt.figure(figsize=(14, 6))
        plt.plot(df['date'], df[f'{plant}_cumulative_hours'], label=fa('ساعت کارکرد تجمعی'))

        max_hours = df[f'{plant}_cumulative_hours'].max()

        # محاسبه خطوط تکراری برای هر بازه تا max_hours
        all_lines = set()
        for level in maintenance_levels:
            interval = level['interval']
            multiples = np.arange(interval, max_hours + interval, interval)
            for m in multiples:
                all_lines.add(m)
        
        # حذف خطوط تکراری با اولویت: اگر عددی در بازه بزرگتر هست، خط کوچکتر حذف شود
        final_lines = {}
        
        for level in maintenance_levels:
            interval = level['interval']
            color = level['color']
            multiples = np.arange(interval, max_hours + interval, interval)
            for m in multiples:
                # اگر خط m هنوز ثبت نشده است، ثبت کن با رنگ و سطح این بازه
                if m not in final_lines:
                    final_lines[m] = {'interval': interval, 'color': color}
        
        for line_y, info in sorted(final_lines.items()):
            plt.axhline(
                y=line_y,
                linestyle='--',
                color=info['color'],
                alpha=0.5,             # شفافیت کمتر = کم‌رنگ‌تر
                linewidth=1,           # ضخامت خط کمتر
                label=fa(f'تعمیر {info["interval"]} ساعت')
            )

        plt.title(fa(f'برنامه تعمیرات {plant}'))
        plt.xlabel(fa('تاریخ'))
        plt.ylabel(fa('ساعت کارکرد تجمعی'))

        # تنظیم نمایش محور افقی با فاصله زمانی مناسب (مثلاً هر ماه یک تیک)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        # حالا برچسب‌های محور x با تاریخ شمسی ولی به صورت خوانا (هر چند روز یکبار)
        step = max(1, len(df)//10)  # مثلا هر 10% داده یک تیک بزن
        
        plt.xticks(
            ticks=df['date'].iloc[::step],
            labels=df['jalali_date'].iloc[::step],
            rotation=45
        )
        plt.ylim(0,max_hours+100)
        #plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f'{plant}_hour.png', dpi=300)
        plt.show()
        plt.close()

#پیش بینی نیاز حیاتی با رگرسیون جنگل تصادفی

"""پیش‌بینی مصرف با مدل ماشین لرنینگ"""
print("پیش‌بینی مصرف حیاتی با مدل ML...")
        
# آماده‌سازی داده‌ها
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['date'].dt.dayofweek.isin([4, 5])  # جمعه و پنجشنبه
        
# ویژگی‌ها
features = ['month', 'day_of_year', 'day_of_week', 'is_weekend']
if 'mean_temperature_2m' in df.columns:
    features.append('mean_temperature_2m')
    
if 'mean_relativehumidity_2m' in df.columns:
    features.append('mean_relativehumidity_2m')
    
X = df[features]
y = df['critical_demand'] #جهت برآورد بیشتر با توجه به هدف مسئله
        
# آموزش مدل
model_critical = RandomForestRegressor(n_estimators=400, random_state=42)
model_critical.fit(X, y)
        
# پیش‌بینی روی داده آزمون
y_pred_test = model_critical.predict(X)
        
# محاسبه معیارهای خطا
mae = mean_absolute_error(y, y_pred_test)
rmse = np.sqrt(mean_squared_error(y, y_pred_test))
r2 = r2_score(y, y_pred_test)
        
print(f"📈 ارزیابی مدل روی داده آزمون:")
print(f"   MAE: {mae:.2f}")
print(f"   RMSE: {rmse:.2f}")
print(f"   R²: {r2:.4f}")

#پیش بینی دما و رطوبت
def generate_future_weather(start_date, days_ahead=365):
        """
        تولید دما/رطوبت آینده براساس الگوی تاریخی + نویز با در نظرگیری الگوی فصلی
        """
        future = []
        
        # ایجاد ویژگی‌های زمانی از داده‌های تاریخی
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        
        # محاسبه میانگین و انحراف معیار برای هر روز از سال
        daily_stats = df.groupby('day_of_year').agg({
            'mean_temperature_2m': ['mean', 'std'],
            'mean_relativehumidity_2m': ['mean', 'std']
        }).reset_index()
        
        daily_stats.columns = ['day_of_year', 'temp_mean', 'temp_std', 'hum_mean', 'hum_std']
        
        # پر کردن مقادیر NaN در انحراف معیار (برای روزهایی که فقط یک داده دارند)
        daily_stats['temp_std'] = daily_stats['temp_std'].fillna(daily_stats['temp_std'].mean())
        daily_stats['hum_std'] = daily_stats['hum_std'].fillna(daily_stats['hum_std'].mean())
        
        for i in range(1, days_ahead + 1):
            d = start_date + timedelta(days=i)
            day_of_year = d.timetuple().tm_yday
            
            # یافتن آمار مربوط به این روز از سال
            day_stats = daily_stats[daily_stats['day_of_year'] == day_of_year]
            
            if len(day_stats) > 0:
                temp_mean = day_stats['temp_mean'].iloc[0]
                temp_std = day_stats['temp_std'].iloc[0]
                hum_mean = day_stats['hum_mean'].iloc[0]
                hum_std = day_stats['hum_std'].iloc[0]
                
                # تولید دما و رطوبت با در نظرگیری نویز
                temp = np.random.normal(temp_mean, temp_std * 0.1)  # کاهش واریانس برای پیش‌بینی
                hum = np.random.normal(hum_mean, hum_std * 0.1)
                
                # محدود کردن مقادیر به رنج معقول
                temp = max(-10, min(45, temp))  # محدوده دمای معقول
                hum = max(10, min(100, hum))    # محدوده رطوبت معقول

            # تولید flags روز کاری
            wd = d.weekday()  # Monday=0 ... Sunday=6
            is_thursday = 1 if wd == 3 else 0   # پنجشنبه
            is_holiday = 1 if wd == 4 else 0    # جمعه
            is_working_day = 0 if (is_thursday == 1 or is_holiday == 1) else 1
            
            future.append({
                'date': d,
                'mean_temperature_2m': float(temp),
                'mean_relativehumidity_2m': float(hum),
                'is_working_day': int(is_working_day),
                'is_thursday': int(is_thursday),
                'is_holiday': int(is_holiday)
            })
        
        future_df = pd.DataFrame(future)
        return future

# پیش‌بینی برای 365 روز آینده
days_ahead = 365
current_date = date.today()
predictions_critical = []
predictions_residential = [] 
dates = []

future_weather = generate_future_weather(current_date, days_ahead)

for i in range(days_ahead):
    future_date = current_date + timedelta(days=i)

    # ویژگی‌های تاریخی برای روز i‌ام آینده
    future_features = {
        'month': future_date.month,
        'day_of_year': future_date.timetuple().tm_yday,
        'day_of_week': future_date.weekday(),
        'is_weekend': future_date.weekday() in [3, 4],  # پنج‌شنبه و جمعه
        'mean_temperature_2m': future_weather[i]['mean_temperature_2m'],
        'mean_relativehumidity_2m': future_weather[i]['mean_relativehumidity_2m']
    }

    # ساخت دیتافریم با یک سطر برای پیش‌بینی
    future_df = pd.DataFrame([future_features])
    # پیش‌بینی با مدل
    prediction = model_critical.predict(future_df)[0]
    predictions_critical.append(prediction)
    
    predictions = model_residential_rf.predict(future_df)[0]
    predictions_residential.append(predictions)
    
    dates.append(future_date)
    
# ساخت دیتافریم نهایی برای ذخیره یا رسم
forecast_df = pd.DataFrame({
    'date': dates,
    'predicted_critical_demand': predictions_critical,
    'predicted_residential_demand': predictions_residential
})
# تبدیل تاریخ میلادی به شمسی و اضافه به دیتافریم پیش‌بینی
forecast_df['jalali_date'] = forecast_df['date'].apply(lambda d: jdatetime.date.fromgregorian(date=d).strftime('%Y-%m-%d'))
forecast_df['jalali_year'] = forecast_df['date'].apply(lambda x: jdatetime.date.fromgregorian(date=x).year)
forecast_df['jalali_month'] = forecast_df['date'].apply(lambda x: jdatetime.date.fromgregorian(date=x).month)
forecast_df['jalali_day'] = forecast_df['date'].apply(lambda x: jdatetime.date.fromgregorian(date=x).day)

# نمایش مصرف حیاتی
plt.figure(figsize=(12, 6))
plt.plot(forecast_df['date'], forecast_df['predicted_critical_demand'], label=fa('پیش بینی نیاز حیاتی'), color='purple')
plt.plot(forecast_df['date'], forecast_df['predicted_residential_demand'], label=fa('پیش بینی نیاز مسکونی'), color='blue')
plt.plot(forecast_df['date'], forecast_df['predicted_residential_demand']+forecast_df['predicted_critical_demand'], label=fa('پیش بینی نیاز کل'), color='green')
plt.xlabel(fa('تاریخ'))
plt.ylabel(fa('نیاز پیش بینی شده(MW)'))
plt.title(fa('پیش بینی نیاز حیاتی و مسکونی برای 365 روز آینده'))
plt.grid(linestyle='--', alpha=0.7)
# تنظیم نمایش محور افقی به صورت ماهانه برای خوانایی بهتر
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# تنظیم تیک‌های محور x با فاصله مناسب
step = max(1, days_ahead // 20)  # مثلا هر 5% داده یک تیک
tick_locs = forecast_df['date'].iloc[::step]
tick_labels = forecast_df['jalali_date'].iloc[::step]

plt.xticks(ticks=tick_locs, labels=tick_labels, rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig('predictions_demand.png', dpi=300)
plt.show()
plt.close()

# نمایش نمودار شرایط جوی
future_weather_df = pd.DataFrame(future_weather)
plt.figure(figsize=(12, 6))
plt.plot(future_weather_df['date'], future_weather_df['mean_temperature_2m'], label=fa('دمای متوسط'), color='red')
plt.plot(future_weather_df['date'], future_weather_df['mean_relativehumidity_2m'], label=fa('رطوبت متوسط'), color='blue')
plt.title(fa(f'پیش‌بینی دما و رطوبت برای {days_ahead} روز آینده'))
plt.xlabel(fa('تاریخ'))
plt.ylabel(fa('مقدار'))
step = max(1, days_ahead // 20)  # مثلا هر 5% داده یک تیک
tick_locs = forecast_df['date'].iloc[::step]
tick_labels = forecast_df['jalali_date'].iloc[::step]
plt.xticks(ticks=tick_locs, labels=tick_labels, rotation=90)
plt.show()
plt.close()

cutoff_threshold = 5
n_periods = 36  # تعداد بازه‌های 10 روزه
block_size_days = 10

# استخراج تاریخ‌های میلادی از forecast_df
dates_gregorian = forecast_df['date']
start_date_gregorian = dates_gregorian.iloc[0]
end_date_gregorian = dates_gregorian.iloc[-1]

print(f"تاریخ شروع پیش‌بینی (میلادی): {start_date_gregorian}")
print(f"تاریخ پایان پیش‌بینی (میلادی): {end_date_gregorian}")

# تبدیل تاریخ شروع به شمسی
start_date_jalali = jdatetime.date.fromgregorian(date=start_date_gregorian)
print(f"تاریخ شروع پیش‌بینی (شمسی): {start_date_jalali}")

# محاسبه ریسک برای هر بازه 10 روزه بر اساس الگوی تاریخی
avg_risks = []

for period_idx in range(n_periods):
    # محاسبه تاریخ شروع و پایان این بازه
    period_start = start_date_gregorian + timedelta(days=period_idx * block_size_days)
    period_end = period_start + timedelta(days=block_size_days - 1)
    
    # تبدیل به تاریخ شمسی برای هر روز در این بازه
    period_risks = []
    
    current_date = period_start
    while current_date <= min(period_end, end_date_gregorian):
        # تبدیل به تاریخ شمسی
        current_jalali = jdatetime.date.fromgregorian(date=current_date)
        month_jalali = current_jalali.month
        day_jalali = current_jalali.day
        
        # پیدا کردن داده تاریخی برای این روز شمسی (بدون در نظر گرفتن سال)
        mask = (
            (df['jalali_month'] == month_jalali) & 
            (df['jalali_day'] == day_jalali)
        )
        
        day_data = df.loc[mask, 'Tavanir']
        
        if len(day_data) > 0:
            # محاسبه ریسک برای این روز
            risk = (day_data <= cutoff_threshold).mean()
            period_risks.append(risk)
        
        current_date += timedelta(days=1)
    
    # میانگین ریسک برای این بازه
    if len(period_risks) > 0:
        avg_period_risk = np.mean(period_risks)
    
    avg_risks.append(avg_period_risk)
    
    print(f"بازه {period_idx+1}: {period_start} تا {min(period_end, end_date_gregorian)} - ریسک: {avg_period_risk:.3f}")

print(f"\nمیانگین ریسک‌ها برای {len(avg_risks)} بازه: {avg_risks}")

# اضافه کردن تاریخ شمسی به forecast_df برای استفاده در ادامه
forecast_df['jalali_year'] = forecast_df['date'].apply(lambda x: jdatetime.date.fromgregorian(date=x).year)
forecast_df['jalali_month'] = forecast_df['date'].apply(lambda x: jdatetime.date.fromgregorian(date=x).month)
forecast_df['jalali_day'] = forecast_df['date'].apply(lambda x: jdatetime.date.fromgregorian(date=x).day)

# ----------------------------
# تنظیمات پایه
# ----------------------------
n_days = 365
n_periods = 36  # هر بلاک = 10 روز (36 * 10 = 360؛ آخرین بلاک کوتاه‌تر خواهد بود)
BLOCK_DAYS = 10
HOURS_PER_DAY = 24

# پارامترهای تعمیرات (فرض: دومین مقدار روز است؛ اگر ساعت است آن را به روز تبدیل کن)
MAINTENANCE_SPECS = [
    (500, 1, "Minor500"),
    (2000, 2, "Intermediate2000"),
    (4000, 5, "Major A4000"),
    (8000, 12, "Major B8000"),
    (16000, 90, "Overhaul16000")
]
TOLERANCE_HOURS = 23

# محدودیت‌های عملیاتی
MAX_CONSECUTIVE_DAYS = 21
MIN_CONSECUTIVE_DAYS = 10
MAX_ONLINE_PLANTS = 3

# پارامتر dispatch
MIN_PLANT_UTILIZATION = 0.25  # اگر تخصیص < 25% ظرفیت، از روشن شدن واحد اجتناب کن (اگر ممکن باشد)

# ----------------------------
# پارامترهای نیروگاه‌ها
# ----------------------------
fuel_coefficients = {
    'TG B': [-1.3617, 704, -4991],
    'TG C': [-1.3617, 704, -4991],
    'TG D': [-1.3617, 704, -4991],
    'TG 5': [-0.1966, 258.87, 49424],
    'TG 6': [-0.1966, 258.87, 49424],
    'TG 7': [-0.1966, 258.87, 49424]
}

capacities = {
    'TG B': 160,
    'TG C': 160,
    'TG D': 160,
    'TG 5': 390,
    'TG 6': 390,
    'TG 7': 390
}

initial_hours = {
    'TG B': 16500,
    'TG C': 15300,
    'TG D': 26500,
    'TG 5': 31500,
    'TG 6': 29500,
    'TG 7': 36000
}

plants = list(capacities.keys())

# ----------------------------
# داده‌های بار و ریسک (فرض می‌شود forecast_df از بیرون آمده و در محیط تعریف است)
# کاربر: اگر forecast_df موجود نیست، باید آن را قبل از اجرای این سل وارد کند.
# ----------------------------
try:
    predicted_critical_load_daily = forecast_df['predicted_critical_demand'].values[:n_days]
    predicted_residential_load_daily = forecast_df['predicted_residential_demand'].values[:n_days]
    predicted_total_load_daily = predicted_critical_load_daily + predicted_residential_load_daily
except Exception as e:
    raise RuntimeError("فایل forecast_df در محیط وجود ندارد یا ستون‌های مورد انتظار را ندارد. ابتدا forecast_df را بارگذاری کن.") from e

# تبدیل به بلاک‌های 10 روزه
def aggregate_to_blocks(daily_array):
    blocks = []
    for b in range(n_periods):
        start = b * BLOCK_DAYS
        end = min(n_days, (b+1) * BLOCK_DAYS)
        if start >= len(daily_array):
            blocks.append(0.0)
        else:
            blocks.append(np.mean(daily_array[start:end]))
    return np.array(blocks)

critical_load_block = aggregate_to_blocks(predicted_critical_load_daily)
total_load_block = aggregate_to_blocks(predicted_total_load_daily)

# استفاده از ریسک واقعی (مقداردهی شما)
avg_risks_block = np.array([
    0.09090909090909091, 0.09090909090909091, 0.01818181818181818, 0.0, 0.0,
    0.00909090909090909, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03636363636363636,
    0.01818181818181818, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091,
    0.12727272727272726, 0.18181818181818182, 0.13636363636363635, 0.09090909090909091,
    0.09090909090909091, 0.1181818181818182, 0.2545454545454545, 0.27272727272727265,
    0.27272727272727265, 0.22727272727272724, 0.27272727272727265, 0.27272727272727265,
    0.18181818181818182, 0.19818181818181818, 0.2, 0.17, 0.09272727272727274,
    0.09090909090909091
])
print(f"📊 ریسک توانیر واقعی برای {len(avg_risks_block)} بازه ۱۰ روزه بارگیری شد")

# ----------------------------
# توابع محاسباتی
# ----------------------------
def calculate_fuel_consumption(plant, power_output):
    """محاسبه مصرف سوخت بر اساس مدل درجه دو (a x^2 + b x + c)"""
    if power_output <= 0:
        return 0.0
    a, b, c = fuel_coefficients[plant]
    return max(0.0, a * (power_output ** 2) + b * power_output + c)

def hours_to_days(h):
    return math.ceil(h / 24)

def next_maintenance_days(initial_hours_val, interval_hours):
    hours_since = initial_hours_val % interval_hours
    hours_to_next = interval_hours - hours_since
    return hours_to_days(hours_to_next)

def days_to_blocks(days):
    return math.ceil(days / BLOCK_DAYS)

MIN_CONSECUTIVE_BLOCKS = math.ceil(MIN_CONSECUTIVE_DAYS / BLOCK_DAYS)
MAX_CONSECUTIVE_BLOCKS = math.ceil(MAX_CONSECUTIVE_DAYS / BLOCK_DAYS)

# ----------------------------
# جایگزین risk_based_dispatch و create_initial_solution_with_maintenance
# ----------------------------

def _duration_value_to_days(duration_value):
    """
    اگر duration_value بزرگ باشد (فرض > 48) آن را ساعت در نظر می‌گیریم و به روز تبدیل می‌کنیم.
    در غیر اینصورت آن را روز فرض می‌کنیم.
    (این یک قاعدهٔ محافظه‌کارانه برای تطبیق با ورودی‌های متفاوت است.)
    """
    if duration_value > 48:
        return math.ceil(duration_value / 24)
    else:
        return int(max(1, duration_value))

def risk_based_dispatch(available_plants, demand, risk_level):
    """
    نسخه‌ی بهبود یافته dispatch:
    - اگر ریسک < 0.05: تمام تقاضا از توانیر (dispatch = {})
    - اگر ریسک >= 0.2: سختگیرانه — همهٔ واحدهای آنلاین تا حد ظرفیت پر می‌شوند (بدون رد تخصیص‌های کوچک)
    - اگر 0.05 <= ریسک < 0.2: سیاست ملایم با MIN_PLANT_UTILIZATION
    """
    if not available_plants:
        return {}, demand

    if risk_level < 0.05:
        return {}, demand

    # ترتیب بر اساس کارایی (کمترین مصرف در سطح مرجع اول)
    efficient_plants = sorted(available_plants, key=lambda p: calculate_fuel_consumption(p, capacities[p] * 0.8))

    dispatch = {}
    remaining = float(demand)

    if risk_level >= 0.2:
        # سختگیرانه: پر کردن تا ظرفیت واحدها (بدون شرط MIN_PLANT_UTILIZATION)
        for plant in efficient_plants:
            if remaining <= 1e-6:
                break
            cap = capacities[plant]
            allocate = min(cap, remaining)
            if allocate > 0:
                dispatch[plant] = allocate
                remaining -= allocate
        tavanir_import = max(0.0, remaining)
        return dispatch, tavanir_import

    # حالا حالت میانه (0.05 <= risk < 0.2) — با MIN_PLANT_UTILIZATION
    for plant in efficient_plants:
        if remaining <= 1e-6:
            break
        cap = capacities[plant]
        allocate = min(cap, remaining)
        if allocate < MIN_PLANT_UTILIZATION * cap:
            other_caps = sum(capacities[p] for p in efficient_plants if p != plant)
            if other_caps < remaining:
                dispatch[plant] = allocate
                remaining -= allocate
            else:
                continue
        else:
            dispatch[plant] = allocate
            remaining -= allocate

    tavanir_import = max(0.0, remaining)
    return dispatch, tavanir_import


def create_initial_solution_with_maintenance():
    """
    برنامه‌ریزی تمام تعمیرات در طول horizon براساس initial_hours و MAINTENANCE_SPECS.
    خروجی: sol (n_plants x n_periods) با مقادیر 0=online, 1=offline
    """
    n_plants = len(plants)
    sol = np.zeros((n_plants, n_periods), dtype=int)  # 0 = online, 1 = offline

    # ما برای هر نیروگاه شبیه‌سازی می‌کنیم: از initial_hours تا انتهای horizon
    horizon_hours = n_days * 24

    maintenance_calendar = {p: [] for p in plants}  # برای debug: لیست (start_day, duration_days, name)

    for i, plant in enumerate(plants):
        current_hours = initial_hours[plant]
        # برای هر spec، برنامه‌ریزی تکراری تعمیرها که در horizon میفتند
        for interval_hours, duration_value, name in MAINTENANCE_SPECS:
            # دوره را تا زمانی که رخداد بعدی داخل horizon است اضافه کن
            # محاسبه اولین فاصله تا رخداد بعدی
            hours_to_next = interval_hours - (current_hours % interval_hours)
            # اگر hours_to_next == interval_hours و current_hours % interval_hours ==0
            if hours_to_next == interval_hours:
                hours_to_next = 0  # means it's due now

            # iterate occurrences of this spec within horizon
            occ = hours_to_next
            while occ < horizon_hours:
                # تعیین روز شروع
                start_day = int(math.floor(occ / 24.0))
                # تبدیل duration_value به روز
                dur_days = _duration_value_to_days(duration_value)
                # تعیین بلاک‌ها متناظر
                start_block = start_day // BLOCK_DAYS
                duration_blocks = max(1, days_to_blocks(dur_days))
                end_block = min(n_periods - 1, start_block + duration_blocks - 1)

                # علامت‌گذاری آفلاین در sol (اضافه — چون ممکن است چند spec روی هم بیفتد)
                sol[i, start_block:end_block + 1] = 1

                # ذخیره برای دیباگ
                maintenance_calendar[plant].append((start_day, dur_days, name))

                # رفتن به رخداد بعدی همین نوع سرویس
                occ += interval_hours

        # توجه: ممکن است چندین نوع سرویس (spec) در سال تکرار شوند و روی هم بیفتند — sol با 1 پوشش می‌دهد.

    # نمایش تقویم تعمیرات برای هر نیروگاه (debug)
    print("=== تقویم تعمیرات برنامه‌ریزی شده (شروع-روز، مدت(روز)، نام) ===")
    for p in plants:
        lst = maintenance_calendar[p]
        print(p, ":", lst)

    # سپس enforce MIN/MAX پیاپی (بر حسب بلاک) همانند قبل
    def enforce_min_max_runs_on_row(row):
        r = row.copy()
        j = 0
        L = len(r)
        while j < L:
            val = r[j]
            k = j
            while k < L and r[k] == val:
                k += 1
            run_len = k - j
            if val == 0:  # online run
                if run_len < MIN_CONSECUTIVE_BLOCKS:
                    need = MIN_CONSECUTIVE_BLOCKS - run_len
                    # کوشش برای تبدیل بلوک‌های بعدی به offline
                    r[k:k+need] = 1
                elif run_len > MAX_CONSECUTIVE_BLOCKS:
                    excess = run_len - MAX_CONSECUTIVE_BLOCKS
                    r[k-excess:k] = 1
            else:  # offline run
                if run_len < MIN_CONSECUTIVE_BLOCKS:
                    need = MIN_CONSECUTIVE_BLOCKS - run_len
                    r[k:k+need] = 0
                elif run_len > MAX_CONSECUTIVE_BLOCKS:
                    excess = run_len - MAX_CONSECUTIVE_BLOCKS
                    r[k-excess:k] = 0
            j = k
        return r

    for i in range(n_plants):
        sol[i, :] = enforce_min_max_runs_on_row(sol[i, :])

    # محدود کردن تعداد آنلاین در هر بلاک تا MAX_ONLINE_PLANTS (حریصانه - حذف کم‌بهره‌ها)
    plant_eff_rank = sorted(plants, key=lambda p: calculate_fuel_consumption(p, capacities[p]*0.8), reverse=True)
    for b in range(n_periods):
        online_idx = [idx for idx, p in enumerate(plants) if sol[idx, b] == 0]
        while len(online_idx) > MAX_ONLINE_PLANTS:
            removed = None
            for p in plant_eff_rank:
                idx = plants.index(p)
                if idx in online_idx:
                    sol[idx, b] = 1
                    removed = idx
                    break
            if removed is None:
                break
            online_idx = [idx for idx, p in enumerate(plants) if sol[idx, b] == 0]

    # پاس نهایی MIN/MAX مجدد
    for i in range(n_plants):
        sol[i, :] = enforce_min_max_runs_on_row(sol[i, :])

    return sol

# ----------------------------
# تبدیل بلاک -> روز و enforce روزانهٔ MAX_ONLINE
# ----------------------------
def enforce_daily_max_online_from_blocks(best_sol_blocks):
    # تبدیل به روزانه
    daily = {}
    for i, plant in enumerate(plants):
        arr = np.zeros(n_days, dtype=int)
        for b in range(n_periods):
            start = b * BLOCK_DAYS
            end = min(n_days, (b+1) * BLOCK_DAYS)
            arr[start:end] = best_sol_blocks[i, b]
        daily[plant] = arr

    # پاس روزانه: اگر بیش از MAX_ONLINE_PLANTS آنلاین باشند، برخی را آفلاین کن
    for day in range(n_days):
        online_plants = [p for p in plants if daily[p][day] == 0]
        if len(online_plants) <= MAX_ONLINE_PLANTS:
            continue
        ranked = sorted(online_plants, key=lambda p: calculate_fuel_consumption(p, capacities[p]*0.8), reverse=True)
        to_remove = len(online_plants) - MAX_ONLINE_PLANTS
        for j in range(to_remove):
            p = ranked[j]
            daily[p][day] = 1

    return daily

# ----------------------------
# اجرای ساخت برنامه و تبدیل نهایی
# ----------------------------
print("🔧 شروع بهینه‌سازی زمان‌بندی نیروگاه‌ها...")
best_solution = create_initial_solution_with_maintenance()

# اگر ریسک پایین در بلاکی وجود دارد، آن بلاک را روی آفلاین تنظیم کن (طبق خواست شما)
for b in range(n_periods):
    if b < len(avg_risks_block) and avg_risks_block[b] < 0.05:
        best_solution[:, b] = 1

# چاپ آماری بلاک‌ها
block_online_counts = [int(sum(1 for i in range(len(plants)) if best_solution[i,b]==0)) for b in range(n_periods)]
print("تعداد نیروگاه‌های آنلاین در هر بلاک (پس از اعمال maintenance و ریسک پایین):")
print(block_online_counts)
print("حداکثر آنلاین در بلاک‌ها: " + str(max(block_online_counts)))

# تبدیل بلاک->روز و enforce روزانه
final_schedule = enforce_daily_max_online_from_blocks(best_solution)
for p in plants:
    final_schedule[p] = np.asarray(final_schedule[p], dtype=int)

# نمونهٔ ده روز اول برای بررسی
print("نمونه: تعداد نیروگاه‌های آنلاین روز اول تا ده روز اول:")
print([sum(1 for p in plants if final_schedule[p][d] == 0) for d in range(10)])

# ----------------------------
# شبیه‌سازی روزانه (حلقهٔ اصلی) — اینجا dispatch و کنترل‌ها اجرا می‌شود
# ----------------------------
daily_operations = []
daily_generation = []
daily_import = []
daily_fuel_consumption = []
daily_online_count = []
daily_risk = []

for day in range(n_days):
    # تهیه لیست نیروگاه‌های آنلاین بر اساس final_schedule
    online_plants = [p for p in plants if final_schedule[p][day] == 0]
    daily_online_count.append(len(online_plants))

    # index بلوک برای تعیین ریسک
    block_idx = day // BLOCK_DAYS
    risk_level = avg_risks_block[block_idx] if block_idx < len(avg_risks_block) else 0.1
    daily_risk.append(risk_level)

    # فراخوانی dispatch
    dispatch, tavanir_import = risk_based_dispatch(online_plants, predicted_total_load_daily[day], risk_level)

    # اگر به هر دلیل اختصاص > تقاضا شد، مقیاس بزن
    total_alloc = sum(dispatch.values())
    demand = float(predicted_total_load_daily[day])
    if total_alloc > demand + 1e-6:
        scale = demand / total_alloc
        for k in list(dispatch.keys()):
            dispatch[k] = dispatch[k] * scale
        tavanir_import = 0.0

    # هشدار سریع اگر هنوز بیش از حد آنلاین است (باید کم شود ولی برای گزارش)
    if sum(1 for p in plants if final_schedule[p][day] == 0) > MAX_ONLINE_PLANTS:
        print(f"⚠️ هشدار: روز {day} تعداد نیروگاه‌های آنلاین بیشتر از حد مجاز است: " +
              str(sum(1 for p in plants if final_schedule[p][day] == 0)))

    # محاسبه مصرف سوخت
    total_fuel = 0.0
    for plant, power in dispatch.items():
        total_fuel += calculate_fuel_consumption(plant, power)

    daily_generation.append(sum(dispatch.values()))
    daily_import.append(tavanir_import)
    daily_fuel_consumption.append(total_fuel)

    # ثبت عملیات روزانه برای گانت و آمار
    for plant in plants:
        status = 'online' if final_schedule[plant][day] == 0 else 'offline'
        daily_operations.append({
            'day': day,
            'plant': plant,
            'status': status,
            'generation': dispatch.get(plant, 0.0),
            'risk_level': risk_level
        })

# DataFrame خروجی
df_operations = pd.DataFrame(daily_operations)

# ----------------------------
# رسم نمودارها (Matplotlib) — فارسی‌شده با fa()
# ----------------------------
print("📈 رسم نمودارهای تحلیلی...")

# نمودار ۱: تامین بار و تولید
plt.figure(figsize=(16, 12))
plt.subplot(4, 1, 1)
plt.plot(range(n_days), predicted_total_load_daily, label=fa('بار کل'), linewidth=1.5)
plt.plot(range(n_days), predicted_critical_load_daily, label=fa('بار بحرانی'), linewidth=1.5)
plt.plot(range(n_days), daily_generation, label=fa('تولید نیروگاه‌ها'), linewidth=1.5)
plt.plot(range(n_days), daily_import, label=fa('واردات از توانیر'), linewidth=1.5)
plt.ylabel(fa('توان (مگاوات)'))
plt.title(fa('تامین بار در طول سال - با استفاده از ریسک واقعی توانیر'))
plt.legend()
plt.grid(True, alpha=0.3)

# نمودار ریسک
plt.subplot(4, 1, 2)
plt.plot(range(n_days), daily_risk, label=fa('ریسک توانیر'), linewidth=2)
plt.axhline(y=0.05, linestyle='--', alpha=0.7, label=fa('آستانه ریسک پایین (0.05)'))
plt.axhline(y=0.2, linestyle='--', alpha=0.7, label=fa('آستانه ریسک بالا (0.2)'))
plt.ylabel(fa('سطح ریسک'))
plt.legend()
plt.grid(True, alpha=0.3)

# مصرف سوخت
plt.subplot(4, 1, 3)
plt.plot(range(n_days), daily_fuel_consumption, label=fa('مصرف سوخت'), linewidth=1.2)
plt.ylabel(fa('مصرف سوخت'))
plt.xlabel(fa('روز سال'))
plt.legend()
plt.grid(True, alpha=0.3)

# تعداد نیروگاه‌های آنلاین
plt.subplot(4, 1, 4)
plt.plot(range(n_days), daily_online_count, label=fa('تعداد نیروگاه‌های آنلاین'), linewidth=1.2)
plt.axhline(y=MAX_ONLINE_PLANTS, linestyle='--', alpha=0.7, color='red', label=fa(f'حداکثر مجاز ({MAX_ONLINE_PLANTS})'))
plt.ylabel(fa('تعداد نیروگاه‌ها'))
plt.xlabel(fa('روز سال'))
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('power_supply_analysis_real_risk.png', dpi=300, bbox_inches='tight')
plt.show()

# نمودار ۲: گانت چارت وضعیت نیروگاه‌ها
plt.figure(figsize=(16, 10))
colors_online = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
colors_offline = ['#95a5a6', '#7f8c8d', '#34495e', '#2c3e50', '#bdc3c7', '#ecf0f1']

for i, plant in enumerate(plants):
    plant_data = df_operations[df_operations['plant'] == plant].sort_values('day')
    status_blocks = []
    current_status = plant_data.iloc[0]['status']
    start_day = plant_data.iloc[0]['day']
    for idx in range(1, len(plant_data)):
        current_row = plant_data.iloc[idx]
        if current_row['status'] != current_status:
            status_blocks.append((start_day, current_row['day'] - 1, current_status))
            current_status = current_row['status']
            start_day = current_row['day']
    status_blocks.append((start_day, plant_data.iloc[-1]['day'], current_status))

    for start, end, status in status_blocks:
        duration = end - start + 1
        color = colors_online[i] if status == 'online' else colors_offline[i]
        alpha = 0.9 if status == 'online' else 0.6
        plt.barh(i, duration, left=start, height=0.7, color=color, alpha=alpha, edgecolor='white')

plt.yticks(range(len(plants)), [fa(p) for p in plants])
plt.xlabel(fa('روز سال'))
plt.title(fa('نمودار گانت وضعیت نیروگاه‌ها\nرنگ‌های روشن: آنلاین، رنگ‌های تیره: آفلاین/تعمیر'))
plt.grid(True, alpha=0.25)

# راهنما
from matplotlib.patches import Patch
legend_elements = []
for i, plant in enumerate(plants):
    legend_elements.append(Patch(facecolor=colors_online[i], alpha=0.9, label=fa(f'{plant} - آنلاین')))
    legend_elements.append(Patch(facecolor=colors_offline[i], alpha=0.6, label=fa(f'{plant} - آفلاین')))
plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('plant_gantt_chart_detailed.png', dpi=300, bbox_inches='tight')
plt.show()

# نمودار ۳: تحلیل ریسک و استراتژی (منطقه‌ای)
plt.figure(figsize=(14, 6))
plt.plot(range(n_days), daily_risk, label=fa('ریسک توانیر'), linewidth=2)
plt.fill_between(range(n_days), 0, 0.05, alpha=0.15)
plt.fill_between(range(n_days), 0.05, 0.2, alpha=0.12)
plt.fill_between(range(n_days), 0.2, 1.0, alpha=0.08)
plt.xlabel(fa('روز سال'))
plt.ylabel(fa('سطح ریسک'))
plt.title(fa('استراتژی تامین بار بر اساس سطح ریسک واقعی توانیر'))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('risk_analysis_real.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------
# گزارش نهایی
# ----------------------------
print("\n📊 گزارش نهایی بهینه‌سازی با ریسک واقعی:")
total_generation = sum(daily_generation)
total_import = sum(daily_import)
total_fuel = sum(daily_fuel_consumption)
self_sufficiency = total_generation / sum(predicted_total_load_daily) * 100

print("✅ کل تولید داخلی: " + f"{total_generation:,.0f} مگاوات-روز")
print("✅ کل واردات از توانیر: " + f"{total_import:,.0f} مگاوات-روز")
print("✅ کل مصرف سوخت: " + f"{total_fuel:,.0f} واحد")
print("✅ نسبت خودکفایی: " + f"{self_sufficiency:.1f}%")

print(fa("\n📈 آمار وضعیت نیروگاه‌ها:"))
for plant in plants:
    online_days = int(sum(final_schedule[plant] == 0))
    offline_days = int(sum(final_schedule[plant] == 1))
    utilization = online_days / n_days * 100
    plant_data = df_operations[(df_operations['plant'] == plant) & (df_operations['status'] == 'online')]
    plant_generation = plant_data['generation'].mean() if len(plant_data) > 0 else 0
    estimated_hours = initial_hours[plant] + (online_days * 24)

    print(f"   {plant}:")
    print("     🟢 روزهای آنلاین: " + f"{online_days} ({utilization:.1f}%)")
    print("     🔴 روزهای آفلاین/تعمیر: " + f"{offline_days}")
    print("     ⚡ تولید متوسط: " + f"{plant_generation:.0f} مگاوات")
    print("     ⏰ ساعت کارکرد تخمینی: " + f"{estimated_hours:,.0f} ساعت")

low_risk_days = sum(1 for r in daily_risk if r < 0.05)
medium_risk_days = sum(1 for r in daily_risk if 0.05 <= r < 0.2)
high_risk_days = sum(1 for r in daily_risk if r >= 0.2)

print("\n🎯 استراتژی بر اساس ریسک REAL:")
print("   روزهای ریسک پایین: " + f"{low_risk_days} ({low_risk_days/n_days*100:.1f}%)")
print("   روزهای ریسک متوسط: " + f"{medium_risk_days} ({medium_risk_days/n_days*100:.1f}%)")
print("   روزهای ریسک بالا: " + f"{high_risk_days} ({high_risk_days/n_days*100:.1f}%)")

avg_online = np.mean(daily_online_count)
max_online = np.max(daily_online_count)
over_limit_days = sum(1 for count in daily_online_count if count > MAX_ONLINE_PLANTS)

print("\n📋 ارزیابی قیود:")
print("   میانگین نیروگاه‌های آنلاین: " + f"{avg_online:.1f}")
print("   حداکثر نیروگاه‌های آنلاین: " + f"{max_online}")
print("   روزهای بیش از حد مجاز: " + f"{over_limit_days} ({over_limit_days/n_days*100:.1f}%)")

print("\n🔍 تحلیل ریسک واقعی:")
print("   میانگین ریسک سالانه: " + f"{np.mean(avg_risks_block):.3f}")
print("   حداکثر ریسک: " + f"{np.max(avg_risks_block):.3f}")
print("   حداقل ریسک: " + f"{np.min(avg_risks_block):.3f}")

print("\n✅ شبیه‌سازی با ریسک واقعی کامل شد!")
print("📁 فایل‌های خروجی:")
print("   - power_supply_analysis_real_risk.png")
print("   - plant_gantt_chart_detailed.png")
print("   - risk_analysis_real.png")




