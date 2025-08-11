import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import arabic_reshaper
from bidi.algorithm import get_display
import qrcode
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# تنظیمات فارسی‌نویسی
plt.rcParams["font.family"] = "Adobe Arabic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.size'] = 18

# تابع کمکی برای نمایش متن فارسی
def persian_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# تنظیمات اولیه
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

# ========================
# تولید داده‌های شبیه‌سازی شده پالایشگاه
# ========================
print("در حال تولید داده‌های شبیه‌سازی شده...")
n_samples = 1500

data = {
    'دما (°C)': np.random.normal(325, 25, n_samples),
    'فشار (bar)': np.random.uniform(40, 160, n_samples),
    'دبی خوراک (m³/h)': np.random.lognormal(3.2, 0.4, n_samples),
    'غلظت کاتالیست (%)': np.random.beta(2, 5, n_samples) * 12,
    'ناخالصی (ppm)': np.random.gamma(2, 5, n_samples)
}

df = pd.DataFrame(data)

# محاسبه بازدهی (تابع غیرخطی پیچیده)
df['بازدهی (%)'] = (
    0.6 * (df['دما (°C)'] - 300) +
    1.8 * np.log(df['فشار (bar)']) -
    0.05 * (df['دبی خوراک (m³/h)'] - 25)**2 +
    4.5 * np.sqrt(df['غلظت کاتالیست (%)']) -
    0.08 * df['ناخالصی (ppm)'] +
    np.random.normal(0, 4, n_samples)
)

# محدود کردن بازدهی بین 40 تا 95
df['بازدهی (%)'] = np.clip(df['بازدهی (%)'], 40, 95)

# ========================
# آماده‌سازی داده‌ها برای مدل
# ========================
print("آماده‌سازی داده‌ها برای مدل...")
X = df.drop('بازدهی (%)', axis=1)
y = df['بازدهی (%)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# ========================
# آموزش مدل جنگل تصادفی
# ========================
print("آموزش مدل پیش‌بینی بازدهی...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

# ارزیابی مدل
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
print(f"دقت مدل (R²) - داده آموزشی: {r2_score(y_train, train_pred):.3f}")
print(f"دقت مدل (R²) - داده آزمون: {r2_score(y_test, test_pred):.3f}")

# ========================
# نمودار اهمیت ویژگی‌ها
# ========================
print("ترسیم نمودار اهمیت ویژگی‌ها...")
importances = model.feature_importances_
features = X.columns
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], color='#1f77b4')
plt.yticks(range(len(sorted_idx)), [persian_text(f) for f in features[sorted_idx]], fontsize=22)
plt.title(persian_text('اهمیت متغیرها در پیش‌بینی بازدهی محصول'), fontsize=24, pad=20)
plt.xlabel(persian_text('میزان اهمیت'), fontsize=22)
plt.grid(axis='x', alpha=0.5)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

# ========================
# نمودارهای وابستگی جزئی (PDP) - نسخه اصلاح شده
# ========================
print("ترسیم نمودارهای وابستگی جزئی...")
fig, ax = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
features_to_plot = [
    'دما (°C)',
    'فشار (bar)',
    'دبی خوراک (m³/h)',
    'غلظت کاتالیست (%)'
]

for i, feature in enumerate(features_to_plot):
    row, col = i // 2, i % 2

    # محاسبه وابستگی جزئی - ساختار جدید خروجی
    pdp_result = partial_dependence(
        model, X_train, [feature],
        kind='average',
        grid_resolution=50
    )

    # استخراج نتایج از خروجی جدید
    grid_values = pdp_result['grid_values']
    avg_predictions = pdp_result['average']

    # رسم نمودار به صورت دستی
    ax[row, col].plot(
        grid_values[0],
        avg_predictions[0],
        color='darkred',
        linewidth=2.5
    )

    # پر کردن ناحیه اطمینان
    ax[row, col].fill_between(
        grid_values[0],
        avg_predictions[0] - 1.5,
        avg_predictions[0] + 1.5,
        alpha=0.15,
        color='darkred'
    )

    ax[row, col].set_title(persian_text(f'وابستگی جزئی به {feature}'), fontsize=22)
    ax[row, col].set_ylabel(persian_text('تغییرات بازدهی پیش‌بینی شده'), fontsize=20)
    ax[row, col].set_xlabel(persian_text(feature), fontsize=20)

plt.suptitle(persian_text('تأثیر متغیرهای فرآیندی بر بازدهی محصول (وابستگی جزئی)'), fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('partial_dependence.png', dpi=300)
plt.show()

# ========================
# نمودارهای انتظار شرطی فردی (ICE)
# ========================
print("ترسیم نمودارهای انتظار شرطی فردی...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# ICE برای دما
PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    ['دما (°C)'],
    kind='individual',
    ax=ax1,
    line_kw={'color': 'gray', 'alpha': 0.2, 'linewidth': 0.8}
)
ax1.set_title(persian_text('تأثیر دما بر بازدهی (ICE Plots)'), fontsize=23)
ax1.set_ylabel(persian_text('تغییرات بازدهی پیش‌بینی شده'), fontsize=21)
ax1.set_xlabel(persian_text('دما (°C)'), fontsize=21)

# ICE برای فشار
PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    ['فشار (bar)'],
    kind='individual',
    ax=ax2,
    line_kw={'color': 'gray', 'alpha': 0.2, 'linewidth': 0.8}
)
ax2.set_title(persian_text('تأثیر فشار بر بازدهی (ICE Plots)'), fontsize=13)
ax2.set_xlabel(persian_text('فشار (bar)'), fontsize=11)

# ایجاد بارکد
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=5, border=2)
qr.add_data("https://B2n.ir/pu6714")
qr.make(fit=True)

# تبدیل بارکد به تصویر
img = qr.make_image(fill_color="black", back_color="white")
img = img.convert("RGBA")
img_np = np.array(img)

# ایجاد جعبه برای بارکد با تنظیمات صحیح
imagebox = OffsetImage(img_np, zoom=0.8)  
ab = AnnotationBbox(
    imagebox, 
    (0.3, 0.7),  
    xycoords='figure fraction',  # استفاده از مختصات شکل اصلی
    box_alignment=(1, 0), 
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به محور فعلی
ax1.add_artist(ab)

plt.suptitle(persian_text('تحلیل رفتار فردی نمونه‌ها با تغییر متغیرهای فرآیندی'), fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('fig2_25.png', dpi=300)
plt.show()
