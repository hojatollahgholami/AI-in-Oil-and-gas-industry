# !pip install pandas numpy scikit-learn xgboost shap pyGAM alepython pdpbox matplotlib arabic-reshaper python-bidi


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.font_manager as fm


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# توضیح‌پذیری
from pdpbox.pdp import PDPIsolate
import PyALE as ale
from pygam import LinearGAM, s
import shap


# تنظیم فونت فارسی
try:
    plt.rcParams['font.family'] = 'sans-serif'
except:
    # اگر فونت XB Zar وجود نداشت، از فونت پیش‌فرض استفاده می‌کند
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


# تابع برای نمایش متن فارسی
def persian_text(text):
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)
    except:
        return text


# ------------------- 1. بارگذاری و آماده‌سازی داده -------------------
df = pd.read_csv("MER_T04_03.csv")


df["year"] = df["YYYYMM"].astype(str).str[:4].astype(int)
df["month"] = df["YYYYMM"].astype(str).str[4:6].astype(int)


# حذف ردیف‌های سالانه (month=13)
df = df[df["month"] <= 12].copy()
df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")


df = df[["MSN", "Description", "date", "year", "month", "Value", "Unit"]]


# لیست بخش‌ها
sectors = df["Description"].unique()


# پوشه خروجی
os.makedirs("plots", exist_ok=True)


# ------------------- 2. پردازش برای هر بخش -------------------
for sector in sectors:
    df_sec = df[df["Description"]==sector].copy()
    
    # تبدیل مقادیر به عددی و مدیریت مقادیر نامعتبر
    df_sec["Value"] = pd.to_numeric(df_sec["Value"], errors='coerce')
    df_sec = df_sec.dropna(subset=["Value"])
    
    # ساخت فیچرها
    df_sec["lag1"] = df_sec["Value"].shift(1)
    df_sec["lag3"] = df_sec["Value"].shift(3)
    df_sec["rolling_6"] = df_sec["Value"].rolling(6, min_periods=1).mean()
    df_sec = df_sec.dropna()


    if len(df_sec) < 24:  # داده کم باشه مدل‌سازی ممکن نیست
        continue


    X = df_sec[["year","month","lag1","lag3","rolling_6"]]
    y = df_sec["Value"]


    # آموزش مدل
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)


    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=300, max_depth=4, learning_rate=0.05)
    model.fit(X_train_s, y_train)


    # ====================== 3. رسم نمودارها ======================


    # ---- 3.1 PDP ----
    X_train_df = pd.DataFrame(X_train_s, columns=X_train.columns)
    pdp_month = PDPIsolate(
        model=model, 
        df=X_train_df,
        model_features=X_train.columns, 
        feature="month",
        feature_name='month',
        n_classes=0
    )


    fig1, ax1 = pdp_month.plot(engine='matplotlib')
    ax1['title_axes'].set_title(persian_text(f"تأثیر ماه بر مصرف انرژی - {sector}"))
    ax1['title_axes'].set_ylabel(persian_text("تأثیر جزئی"))
    ax1['title_axes'].set_xlabel(persian_text("ماه"))
    fig1.savefig(f"plots/{sector}_PDP.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)


    # ---- 3.2 ALE ----
    X_train_eff = pd.DataFrame(X_train_s, columns=X_train.columns)
    eff = ale.ale(
        X=X_train_eff,
        model=model,
        feature=['month'],
        feature_type='continuous',
        include_CI=True, 
        plot=True
    )


    plt.title(persian_text(f"تأثیر تجمعی ماه بر مصرف انرژی - {sector}"))
    plt.xlabel(persian_text("ماه"))
    plt.ylabel(persian_text("تأثیر تجمعی"))
    plt.legend([persian_text("تأثیر"), persian_text("فاصله اطمینان")])
    plt.savefig(f"plots/{sector}_ALE.png", dpi=150, bbox_inches='tight')
    plt.close()


    # ---- 3.3 GAM ----
    gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4)).fit(X_train.values, y_train)
    fig3, axs = plt.subplots(1, X_train.shape[1], figsize=(20, 6))


    feature_names_persian = [
        persian_text("سال"),
        persian_text("ماه"),
        persian_text("تأخیر یک‌ماهه"),
        persian_text("تأخیر سه‌ماهه"),
        persian_text("میانگین متحرک شش‌ماهه")
    ]


    for i, ax in enumerate(axs):
        XX = gam.generate_X_grid(term=i)
        ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
        ax.set_title(feature_names_persian[i])
        ax.set_xlabel(feature_names_persian[i])
        ax.set_ylabel(persian_text("تأثیر جزئی"))


    plt.suptitle(persian_text(f"تأثیر ویژگی‌ها بر مصرف انرژی - {sector}"))
    plt.tight_layout()
    fig3.savefig(f"plots/{sector}_GAM.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)


    # ---- 3.4 SHAP ----
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test_s)


    feature_names_persian = [
        persian_text("سال"),
        persian_text("ماه"),
        persian_text("تأخیر یک‌ماهه"),
        persian_text("تأخیر سه‌ماهه"),
        persian_text("میانگین متحرک شش‌ماهه")
    ]


    fig4 = plt.figure()
    shap.summary_plot(shap_values, features=X_test, feature_names=feature_names_persian, show=False)
    plt.title(persian_text(f"اهمیت ویژگی‌ها در پیش‌بینی مصرف انرژی - {sector}"))
    plt.tight_layout()
    fig4.savefig(f"plots/{sector}_SHAP.png", dpi=150, bbox_inches='tight')
    plt.close(fig4)


    # ---- نمودارهای تکمیلی ----
    
    # نمودار سری زمانی مصرف انرژی
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    ax5.plot(df_sec["date"], df_sec["Value"])
    ax5.set_title(persian_text(f"سری زمانی مصرف انرژی - {sector}"))
    ax5.set_xlabel(persian_text("تاریخ"))
    ax5.set_ylabel(persian_text("مصرف انرژی"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig5.savefig(f"plots/{sector}_timeseries.png", dpi=150, bbox_inches='tight')
    plt.close(fig5)


    # نمودار اهمیت ویژگی‌های مدل
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model, ax=ax6, height=0.8)
    ax6.set_title(persian_text(f"اهمیت ویژگی‌های مدل - {sector}"))
    ax6.set_yticklabels(feature_names_persian)
    plt.tight_layout()
    fig6.savefig(f"plots/{sector}_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close(fig6)


    # ایجاد گزارش تفسیری
    interpretation_report = f"""
تفسیر مدل برای بخش {sector}:


ویژگی‌های استفاده شده:
1. سال (year): نشان‌دهنده روند بلندمدت تغییرات مصرف انرژی
2. ماه (month): نشان‌دهنده الگوهای فصلی مصرف انرژی
3. تأخیر یک‌ماهه (lag1): مصرف انرژی در ماه قبل
4. تأخیر سه‌ماهه (lag3): مصرف انرژی سه ماه قبل
5. میانگین متحرک شش‌ماهه (rolling_6): میانگین مصرف انرژی در شش ماه گذشته


نتایج:
- نمودار PDP نشان می‌دهد که تغییرات فصلی چگونه بر مصرف انرژی تأثیر می‌گذارد.
- نمودار ALE تأثیر تجمعی هر ماه را بر مصرف انرژی نشان می‌دهد.
- نمودار GAM روابط غیرخطی بین ویژگی‌ها و مصرف انرژی را نشان می‌دهد.
- نمودار SHAP اهمیت هر ویژگی در پیش‌بینی مصرف انرژی را نشان می‌دهد.
"""


    # ذخیره گزارش تفسیری
    with open(f"plots/{sector}_interpretation.txt", "w", encoding="utf-8") as f:
        f.write(interpretation_report)


    print(f"✅ نمودارها و گزارش برای {sector} ذخیره شدند.")
