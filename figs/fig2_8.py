!pip install qrcode[pil] arabic_reshaper python-bidi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import arabic_reshaper
from bidi.algorithm import get_display
import qrcode
from PIL import Image
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# تابع برای نمایش صحیح متون فارسی
def bidi_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# تنظیمات اولیه
plt.rcParams['font.family'] = 'Adobe Arabic' #'Microsoft Uighur'
plt.rcParams['font.size'] = 21
mpl.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

# ایجاد داده‌های مصنوعی برای دمای برج تقطیر
n_samples = 200
base_temp = np.random.normal(80, 5, n_samples)  # دمای پایه حول 80 درجه

# افزودن نقاط پرت (outliers)
outlier_indices = np.random.choice(n_samples, size=10, replace=False)
base_temp[outlier_indices] = np.random.uniform(110, 130, 10)  # نقاط پرت با دمای بالا
outlier_indices = np.random.choice(n_samples, size=8, replace=False)
base_temp[outlier_indices] = np.random.uniform(40, 50, 8)  # نقاط پرت با دمای پایین

# ایجاد دیتافریم
df = pd.DataFrame({
    'زمان': pd.date_range(start='2024-01-01', periods=n_samples, freq='h'),
    'دما': base_temp
})

# محاسبه Z-Score
df['Z-Score'] = np.abs(stats.zscore(df['دما']))

# ایجاد نمودارها در یک گرید 2x2
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# 1. نمودار جعبه‌ای
sns.boxplot(y=df['دما'], ax=axs[0, 0], color='skyblue')
axs[0, 0].set_title(bidi_text('نمودار جعبه‌ای'))
axs[0, 0].set_ylabel(bidi_text('دما'))
axs[0, 0].grid(alpha=0.3)

# افزودن میانگین به نمودار
mean_temp = np.mean(df['دما'])

# 2. هیستوگرام (Histogram)
sns.histplot(df['دما'], bins=20, kde=True, ax=axs[0, 1], color='green')
axs[0, 1].set_title(bidi_text('هیستوگرام توزیع دما'))
axs[0, 1].set_xlabel(bidi_text('دما'))
axs[0, 1].set_ylabel(bidi_text('تعداد نمونه‌ها'))
axs[0, 1].grid(alpha=0.3)

# نشان دادن نقاط پرت در هیستوگرام
lower_outliers = df[df['دما'] < 60]
upper_outliers = df[df['دما'] > 100]
axs[0, 1].scatter(lower_outliers['دما'], [5]*len(lower_outliers),
                 color='red', s=50, alpha=0.7, label=bidi_text('نقاط پرت'))
axs[0, 1].scatter(upper_outliers['دما'], [5]*len(upper_outliers),
                 color='red', s=50, alpha=0.7)
axs[0, 1].axvline(x=60, color='orange', linestyle='--', alpha=0.5)
axs[0, 1].axvline(x=100, color='orange', linestyle='--', alpha=0.5)
axs[0, 1].legend()

# 3. نمودار پراکندگی (Scatter Plot)
axs[1, 0].scatter(df['زمان'], df['دما'], alpha=0.7,
                 c=np.where(df['Z-Score'] > 3, 'red', 'blue'))
axs[1, 0].set_title(bidi_text('نمودار پراکندگی دما بر اساس زمان'))
axs[1, 0].set_xlabel(bidi_text('زمان'))
axs[1, 0].set_ylabel(bidi_text('دما'))
axs[1, 0].tick_params(axis='x', rotation=30)
axs[1, 0].grid(alpha=0.3)
axs[1, 0].axhline(y=mean_temp, color='red', linestyle='--', alpha=0.5)

# افزودن خطوط هشدار
axs[1, 0].axhline(y=100, color='orange', linestyle='--', alpha=0.7)
axs[1, 0].axhline(y=60, color='orange', linestyle='--', alpha=0.7)

# 4. نمودار Z-Score
axs[1, 1].scatter(df['زمان'], df['Z-Score'], alpha=0.7,
                 c=np.where(df['Z-Score'] > 3, 'red', 'green'))
axs[1, 1].set_title(bidi_text('نمودار Z-Score برای تشخیص نقاط پرت'))
axs[1, 1].set_xlabel(bidi_text('زمان'))
axs[1, 1].set_ylabel(bidi_text('مقدار Z-Score'))
axs[1, 1].tick_params(axis='x', rotation=30)
axs[1, 1].grid(alpha=0.3)
axs[1, 1].axhline(y=3, color='red', linestyle='--',
                  label=bidi_text('آستانه Z=3'))
axs[1, 1].legend()

# تنظیم فاصله‌گذاری
#plt.tight_layout(rect=[0, 0, 1, 0.96])  # ایجاد فضای کافی برای عنوان اصلی
# ایجاد بارکد برای لینک 
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L,box_size=5, border=2)
qr.add_data("https://github.com/hojatollahgholami/AI-in-Oil-and-gas-industry/edit/main/figs/fig2_8.py")
qr.make(fit=True)

# تبدیل بارکد به تصویر
img = qr.make_image(fill_color="black", back_color="white")
img = img.convert("RGBA")

# ذخیره موقت بارکد در حافظه
img_np = np.array(img)

# ایجاد جعبه برای بارکد
imagebox = OffsetImage(img_np, zoom=0.5)
ab = AnnotationBbox(imagebox, (0.9, 0.08),
                    xycoords='axes fraction',
                    box_alignment=(7.75, -1.7),
                    frameon=False,
                    pad=0)

# اضافه کردن بارکد به نمودار
axs[0, 1].add_artist(ab)

plt.savefig('fig2-8.png', dpi=300)
plt.show()
