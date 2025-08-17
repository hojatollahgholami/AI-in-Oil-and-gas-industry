import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split
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
np.random.seed(42)

# ========================
# تولید داده‌های شبیه‌سازی شده پالایشگاه
# ========================
n_samples = 1500

data = {
    'دما (C)': np.random.normal(325, 25, n_samples),
    'فشار (bar)': np.random.uniform(40, 160, n_samples),
    'دبی خوراک (m3/h)': np.random.lognormal(3.2, 0.4, n_samples),
    'غلظت کاتالیست (%)': np.random.beta(2, 5, n_samples) * 12,
    'ناخالصی (ppm)': np.random.gamma(2, 5, n_samples)
}

df = pd.DataFrame(data)

# محاسبه بازدهی (تابع غیرخطی پیچیده)
df['بازدهی (%)'] = (
    0.6 * (df['دما (C)'] - 300) +
    1.8 * np.log(df['فشار (bar)']) -
    0.05 * (df['دبی خوراک (m3/h)'] - 25)**2 +
    4.5 * np.sqrt(df['غلظت کاتالیست (%)']) -
    0.08 * df['ناخالصی (ppm)'] +
    np.random.normal(0, 4, n_samples)
)

# محدود کردن بازدهی بین 40 تا 95
df['بازدهی (%)'] = np.clip(df['بازدهی (%)'], 40, 95)

# ========================
# آماده‌سازی داده‌ها برای مدل
# ========================
X = df.drop('بازدهی (%)', axis=1)
y = df['بازدهی (%)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# ========================
# آموزش مدل جنگل تصادفی
# ========================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

# ========================
# نمودار اهمیت ویژگی‌ها
# ========================
importances = model.feature_importances_
features = X.columns
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], color='#1f77b4')
plt.yticks(range(len(sorted_idx)), [persian_text(f) for f in features[sorted_idx]], fontsize=18)
plt.title(persian_text('اهمیت متغیرها در پیش‌بینی بازدهی محصول'), fontsize=20, pad=20)
plt.xlabel(persian_text('میزان اهمیت'), fontsize=18)
plt.grid(axis='x', alpha=0.5)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

# ========================
# نمودارهای وابستگی جزئی (PDP) - نسخه اصلاح شده
# ========================
print("ترسیم نمودارهای وابستگی جزئی...")
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
features_to_plot = [
    'دما (C)',
    'فشار (bar)',
    'دبی خوراک (m3/h)',
    'غلظت کاتالیست (%)'
]

for i, feature in enumerate(features_to_plot):
    ax = axs[i//2, i%2]
    
    # محاسبه PDP با مدیریت خطای KeyError
    result = partial_dependence(
        model,
        X_train,
        [feature],
        grid_resolution=50
    )
    
    # استخراج صحیح نتایج بسته به ساختار خروجی
    if isinstance(result, tuple):
        pdp, axes = result
    else:
        # برای نسخه‌های جدیدتر scikit-learn
        pdp = result['average']
        axes = result['values']
    
    # اطمینان از ساختار آرایه‌ای
    pdp = np.array(pdp).flatten()
    grid_vals = np.array(axes[0] if isinstance(axes, list) else axes).flatten()
    
    # رسم PDP
    ax.plot(grid_vals, pdp, color='darkred', linewidth=2.5)
    
    # پر کردن ناحیه اطمینان
    ax.fill_between(
        grid_vals,
        pdp - 1.5,
        pdp + 1.5,
        alpha=0.15,
        color='darkred'
    )
    
    ax.set_title(persian_text(f'وابستگی جزئی به {feature}'), fontsize=18)
    ax.set_ylabel(persian_text('تغییرات بازدهی پیش‌بینی شده'), fontsize=18)
    ax.set_xlabel(persian_text(feature), fontsize=12)
    
#plt.suptitle(persian_text('تأثیر متغیرهای فرآیندی بر بازدهی محصول'), fontsize=20, y=0.98)
# ایجاد و تنظیم بارکد
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=3,
    border=2
)
qr.add_data("https://B2n.ir/pu6714")
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white")

# تبدیل به تصویر قابل استفاده در matplotlib
img = img.convert("RGBA")
img_np = np.array(img)

# تنظیم موقعیت بارکد (پایین سمت راست)
imagebox = OffsetImage(img_np, zoom=0.8)
ab = AnnotationBbox(
    imagebox, 
    (0.15, 0.75),
    xycoords='figure fraction',
    box_alignment=(1, 0),
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به شکل
fig.add_artist(ab)

plt.tight_layout()
plt.savefig('partial_dependence.png', dpi=300)
plt.show()

# ========================
# نمودارهای انتظار شرطی فردی (ICE) 
# ========================
print("ترسیم نمودارهای انتظار شرطی فردی...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# ICE برای دما
ice_result_temp = partial_dependence(
    model,
    X_train,
    ['دما (C)'],
    kind='individual',
    grid_resolution=50
)

# استخراج صحیح داده‌ها
if isinstance(ice_result_temp, tuple):
    ice_values_temp = ice_result_temp[0]
    grid_temp = ice_result_temp[1][0]
else:
    ice_values_temp = ice_result_temp['individual']
    grid_temp = ice_result_temp['values'][0]

# تبدیل به آرایه NumPy و کاهش ابعاد
ice_array_temp = np.array(ice_values_temp)
if ice_array_temp.ndim == 3:
    ice_array_temp = ice_array_temp[0]

# رسم ICE برای دما (محدود به 100 نمونه برای شفافیت)
for i in range(min(100, ice_array_temp.shape[0])):
    ax1.plot(grid_temp, ice_array_temp[i], color='gray', alpha=0.2, linewidth=0.8)

ax1.set_title(persian_text('تأثیر دما بر بازدهی (ICE)'), fontsize=20)
ax1.set_ylabel(persian_text('تغییرات بازدهی پیش‌بینی شده'), fontsize=18)
ax1.set_xlabel(persian_text('دما (C)'), fontsize=18)

# ICE برای فشار
ice_result_pressure = partial_dependence(
    model,
    X_train,
    ['فشار (bar)'],
    kind='individual',
    grid_resolution=50
)

# استخراج صحیح داده‌ها
if isinstance(ice_result_pressure, tuple):
    ice_values_pressure = ice_result_pressure[0]
    grid_pressure = ice_result_pressure[1][0]
else:
    ice_values_pressure = ice_result_pressure['individual']
    grid_pressure = ice_result_pressure['values'][0]

# تبدیل به آرایه NumPy و کاهش ابعاد
ice_array_pressure = np.array(ice_values_pressure)
if ice_array_pressure.ndim == 3:
    ice_array_pressure = ice_array_pressure[0]

# رسم ICE برای فشار (محدود به 100 نمونه برای شفافیت)
for i in range(min(100, ice_array_pressure.shape[0])):
    ax2.plot(grid_pressure, ice_array_pressure[i], color='gray', alpha=0.2, linewidth=0.8)

ax2.set_title(persian_text('تأثیر فشار بر بازدهی (ICE)'), fontsize=20)
ax2.set_xlabel(persian_text('فشار (bar)'), fontsize=18)

#plt.suptitle(persian_text('تحلیل رفتار فردی نمونه‌ها با تغییر متغیرها'), fontsize=20, y=0.98)
plt.tight_layout()
plt.savefig('ice_plots.png', dpi=300)
plt.show()
