# نصب (در صورت نیاز)
# pip install xgboost shap pyGAM PyALE pdpbox 


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor


# Visualization
import matplotlib.pyplot as plt


# Explainability libs
from pdpbox import pdp, info_plots
import PyALE as ale
from pygam import LinearGAM, s, f
import shap


# ------------ 1. بارگذاری و مرتب‌سازی داده ------------
df = pd.read_csv("MER_T04_03.csv")  
df
# مثالِ reshape: اگر فایل هر بخش یک ستون دارد -> melt به حالت long
# اگر فایلت ستون 'Residential','Commercial',... داره:
if set(['Residential','Commercial','Industrial','Electric Power']).issubset(df.columns):
    df_long = df.melt(id_vars=['year','month'], 
                      value_vars=['Residential','Commercial','Industrial','Electric Power'],
                      var_name='sector', value_name='value')
else:
    df_long = df.copy()


# تبدیل به سری زمانی و ویژگی‌سازی پایه
df_long['date'] = pd.to_datetime(df_long[['year','month']].assign(day=1))
df_long = df_long.sort_values('date').reset_index(drop=True)


# ویژگ‌سازی نمونه (مثلاً هدف: پیش‌بینی مصرف بخش 'Electric Power' ماه بعد)
# فیلتر بخش دلخواه یا ساخت مدل چندبخشی
sector = 'Electric Power'
df_sec = df_long[df_long['sector']==sector].copy()


# ویژگی‌های زمانی
df_sec['month'] = df_sec['date'].dt.month
df_sec['year'] = df_sec['date'].dt.year
df_sec['value_lag1'] = df_sec['value'].shift(1).fillna(method='bfill')
df_sec['rolling_3'] = df_sec['value'].rolling(3, min_periods=1).mean()


# اگر داده‌های آب و هوا یا GDP سراسری داشتی، join کن برای مدل بهتر.


# فرض: X-> ویژگی‌ها، y -> مقدار مصرف همان ماه
X = df_sec[['month','year','value_lag1','rolling_3']].copy()
y = df_sec['value'].values


# تقسیم داده
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# مقیاس‌دهی
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# ------------ 2. آموزش مدل (مثال: XGBoost) ------------
dtrain = xgb.DMatrix(X_train_s, label=y_train)
dtest  = xgb.DMatrix(X_test_s, label=y_test)
params = {"objective":"reg:squarederror", "max_depth":4, "eta":0.1}
bst = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest,'test')], early_stopping_rounds=10, verbose_eval=False)


# برای SHAP به یک wrapper sklearn نیاز داریم یا از TreeExplainer برای XGBoost native
# ------------ 3. PDP (Partial Dependence Plot) ------------
# با pdpbox (نیاز به dataframe اصلی با فیچرهای نام‌دار)
model_sklearn = xgb.XGBRegressor().fit(X_train_s, y_train)  # ساده برای pdpbox
# ایجاد PDP برای feature 'month' (index 0 after scaler mapping - we will use original column names)
pdp_month = pdp.pdp_isolate(model=model_sklearn, dataset=pd.DataFrame(X_train_s, columns=X_train.columns), model_features=X_train.columns, feature='month')
pdp.pdp_plot(pdp_month, 'month')
plt.show()


# ------------ 4. ALE (Accumulated Local Effects) ------------
# alepython expects unscaled original features; pass original train
ale.ale_plot(X_train, model_sklearn, ['month'], include_CI=True)
plt.show()


# ------------ 5. GAM (Generalized Additive Model) ------------
gam = LinearGAM(s(0) + s(1) + s(2) + s(3)).fit(X_train.values, y_train)  # s(i) برای هر فیچر
# نمایش اثر هر تابع جزء
fig, axs = plt.subplots(1, X_train.shape[1], figsize=(15,4))
for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.set_title(f'GAM partial for {X_train.columns[i]}')
plt.tight_layout()
plt.show()


# ------------ 6. SHAP ------------
explainer = shap.Explainer(model_sklearn)   # or shap.TreeExplainer(bst) for raw xgboost
shap_values = explainer(X_test_s)
# summary
shap.summary_plot(shap_values, features=X_test, feature_names=X_test.columns)
# dependence (یکی از فیچرها)
shap.dependence_plot('value_lag1', shap_values.values, X_test, feature_names=X_test.columns)
