pip install qrcode[pil] arabic_reshaper python-bidi

import numpy as np
import matplotlib.pyplot as plt
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display
import qrcode
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# تنظیمات فارسی‌نویسی
plt.rcParams["font.family"] = "B Nazanin"
plt.rcParams["axes.unicode_minus"] = False

def persian_text(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

# تولید داده‌ها
np.random.seed(42)
time = np.arange(0, 35)  # زمان از 0 تا 34

# ایجاد فشار با الگوهای مختلف
pressure = np.concatenate([
    # بخش اول: فشار ثابت در 3 بار با نوسان کم
    np.random.normal(3.0, 0.05, 15),

    # بخش دوم: فشار ثابت در 4 بار با نوسان کم
    np.random.normal(4.0, 0.05, 10),

    # بخش سوم: تغییر مفهومی (شیب افزایشی)
    4.0 + 0.15 * (np.arange(0, 10) + np.random.normal(0, 0.1, 10))
])

# افزودن داده‌های پرت
pressure[5] = 5.2   # داده پرت در بخش اول
pressure[18] = 2.3  # داده پرت در بخش دوم

# تعریف رنگ‌ها و دسته‌ها
colors = []
categories = []
for i in range(len(time)):
    if i < 15:
        colors.append('blue')
        categories.append(persian_text('فشار ثابت ۳ بار'))
    elif i < 25:
        colors.append('green')
        categories.append(persian_text('فشار ثابت ۴ بار'))
    else:
        colors.append('red')
        categories.append(persian_text('تغییر مفهومی'))

# مشخص کردن داده‌های پرت
outliers = [5, 18]
for i in outliers:
    colors[i] = 'orange'
    if i < 15:
        categories[i] = persian_text('داده پرت')
    else:
        categories[i] = persian_text('داده پرت')

# ایجاد نمودار
fig, ax = plt.subplots(figsize=(14, 8))

# رسم نقاط با شکل مکعب
for i in range(len(time)):
    ax.scatter(time[i], pressure[i],
                marker='s', s=100,
                c=colors[i],
                edgecolor='black',
                label=categories[i] if i in [0, 15, 25, 5, 18] else "")

# افزودن خطوط راهنما
ax.axhline(y=3.0, color='blue', linestyle='--', alpha=0.3)
ax.axhline(y=4.0, color='green', linestyle='--', alpha=0.3)

# افزودن خط روند برای بخش شیب دار
slope_x = time[25:]
slope_y = pressure[25:]
ax.plot(slope_x, slope_y, 'r--', alpha=0.5)

# تنظیمات نمودار
ax.set_xlabel(persian_text('زمان (ساعت)'), fontsize=16)
ax.set_ylabel(persian_text('فشار (بار)'), fontsize=16)
ax.grid(alpha=0.2)

# ایجاد راهنما
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(
    unique_labels.values(),
    unique_labels.keys(),
    loc='upper left',
    fontsize=16
)

# افزودن توضیحات

ax.annotate(persian_text('تغییر داده'),
             xy=(15, 4.05), xytext=(12, 4.2),
             arrowprops=dict(facecolor='green', arrowstyle='->'),
             fontsize=16)

ax.annotate(persian_text('تغییر مفهومی'),
             xy=(25, 4.05), xytext=(21, 4.5),
             arrowprops=dict(facecolor='red', arrowstyle='->'),
             fontsize=16)

ax.annotate(persian_text('داده پرت'),
             xy=(5, 5.1), xytext=(7, 5),
             arrowprops=dict(facecolor='orange', arrowstyle='->'),
             fontsize=16,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="orange", alpha=0.8))

# =============================================================================
# ایجاد و اضافه کردن بارکد
# =============================================================================
# ایجاد بارکد برای لینک 
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L,box_size=5, border=2)
qr.add_data("https://github.com/hojatollahgholami/AI-in-Oil-and-gas-industry/edit/main/figs/fig6_2.py")
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
                    box_alignment=(1, 0),
                    frameon=False,
                    pad=0)

# اضافه کردن بارکد به نمودار
ax.add_artist(ab)

# ذخیره و نمایش نمودار
plt.tight_layout()
plt.savefig('pressure_time_scatter_with_qr.png', dpi=300, bbox_inches='tight')
plt.show()
