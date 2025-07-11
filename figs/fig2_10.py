!pip install qrcode[pil] arabic_reshaper python-bidi

import numpy as np
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
import qrcode
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.font_manager as fm

# تابع برای نمایش صحیح متون فارسی
def bidi_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# داده‌های ورودی
features = ['دمای سیال','فشار سیال','لرزش','جریان سیال','رطوبت محیط']
corr = [0.72, 0.65, 0.82, 0.45, 0.33]
wrapper = [0.25, 0.20, 0.30, 0.15, 0.10]
embedded = [0.28, 0.22, 0.25, 0.15, 0.10]

# تبدیل نام ویژگی‌ها به فارسی صحیح
persian_features = [bidi_text(feature) for feature in features]

# تنظیمات نمودار
plt.rcParams['font.family'] = 'B Nazanin'
plt.rcParams['font.size'] = 12
fig, ax = plt.subplots(figsize=(14, 8))  # استفاده از subplot برای کنترل بهتر

# موقعیت‌های میله‌ها روی محور X
x = np.arange(len(persian_features))
width = 0.25

# ایجاد میله‌ها برای هر روش
rects1 = ax.bar(x - width, corr, width, label=bidi_text('روش فیلتر'), color='skyblue', alpha=0.9)
rects2 = ax.bar(x, wrapper, width, label=bidi_text('روش پوشش'), color='salmon', alpha=0.9)
rects3 = ax.bar(x + width, embedded, width, label=bidi_text('روش جاسازی'), color='lightgreen', alpha=0.9)

# تنظیم عنوان و برچسب‌ها
ax.set_xlabel(bidi_text('ویژگی‌ها'), fontsize=21)
ax.set_ylabel(bidi_text('اهمیت ویژگی'), fontsize=21)
ax.set_xticks(x)
ax.set_xticklabels(persian_features, fontsize=21)
ax.legend(fontsize=21)

# افزودن مقادیر عددی روی میله‌ها
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # فاصله 3 نقطه‌ای از بالای میله
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=21)

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

# افزودن خطوط راهنما و گرید
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.0)  # محدوده محور Y از 0 تا 1

# ایجاد بارکد
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=5, border=2)
qr.add_data("https://github.com/hojatollahgholami/AI-in-Oil-and-gas-industry/edit/main/figs/fig2_9.py")
qr.make(fit=True)

# تبدیل بارکد به تصویر
img = qr.make_image(fill_color="black", back_color="white")
img = img.convert("RGBA")
img_np = np.array(img)

# ایجاد جعبه برای بارکد با تنظیمات صحیح
imagebox = OffsetImage(img_np, zoom=0.6)  # کاهش zoom برای اندازه مناسب
ab = AnnotationBbox(
    imagebox, 
    (0.92, 0.5),  # موقعیت در گوشه پایین سمت راست
    xycoords='figure fraction',  # استفاده از مختصات شکل اصلی
    box_alignment=(1, 0),  # تراز در پایین سمت راست
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به محور فعلی
ax.add_artist(ab)

plt.tight_layout()
plt.savefig('fig2-10.png', dpi=300, bbox_inches='tight')
plt.show()
