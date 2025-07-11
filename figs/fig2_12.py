!pip install qrcode[pil] arabic_reshaper python-bidi

import numpy as np
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib as mpl
import qrcode
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# تابع برای نمایش صحیح متون فارسی
def bidi_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# تنظیمات اولیه
plt.rcParams['font.family'] = 'Adobe Arabic' #'B Nazanin'
plt.rcParams['font.size'] = 12
mpl.rcParams['axes.unicode_minus'] = False  # حل مشکل نمایش علامت منفی
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# تولید داده‌ها
x = np.linspace(0, 10, 100)

# 1. رابطه خطی مثبت
y_linear_positive = 2 * x + 3 + np.random.normal(0, 1, len(x))
axs[0, 0].scatter(x, y_linear_positive, color='blue', alpha=0.7)
axs[0, 0].plot(x, 2*x + 3, 'r-', linewidth=2)
axs[0, 0].set_title(bidi_text('رابطه خطی مثبت'), fontsize=21)
axs[0, 0].set_xlabel(bidi_text('ویژگی X'), fontsize=21)
axs[0, 0].set_ylabel(bidi_text('ویژگی Y'), fontsize=21)
axs[0, 0].grid(alpha=0.2)
axs[0, 0].annotate(bidi_text('Y = 2X + 3'), xy=(0.5, 0.9),
                  xycoords='axes fraction', fontsize=21, color='red')

# 2. رابطه خطی منفی
y_linear_negative = -1.5 * x + 15 + np.random.normal(0, 1, len(x))
axs[0, 1].scatter(x, y_linear_negative, color='green', alpha=0.7)
axs[0, 1].plot(x, -1.5*x + 15, 'r-', linewidth=2)
axs[0, 1].set_title(bidi_text('رابطه خطی منفی'), fontsize=21)
axs[0, 1].set_xlabel(bidi_text('ویژگی X'), fontsize=21)
axs[0, 1].set_ylabel(bidi_text('ویژگی Y'), fontsize=21)
axs[0, 1].grid(alpha=0.2)
axs[0, 1].annotate(bidi_text('Y = -1.5X + 15'), xy=(0.5, 0.9),
                  xycoords='axes fraction', fontsize=21, color='red')

# 3. رابطه نمایی
y_exponential = 2 * np.exp(0.5 * x) + np.random.normal(0, 5, len(x))
axs[1, 0].scatter(x, y_exponential, color='purple', alpha=0.7)
axs[1, 0].plot(x, 2 * np.exp(0.5 * x), 'r-', linewidth=2)
axs[1, 0].set_title(bidi_text('رابطه نمایی'), fontsize=21)
axs[1, 0].set_xlabel(bidi_text('ویژگی X'), fontsize=21)
axs[1, 0].set_ylabel(bidi_text('ویژگی Y'), fontsize=21)
axs[1, 0].grid(alpha=0.2)
axs[1, 0].annotate(bidi_text('Y = 2 * e^{0.5X}'), xy=(0.5, 0.9),
                  xycoords='axes fraction', fontsize=21, color='red')

# 4. رابطه سینوسی
y_sinusoidal = 5 * np.sin(x) + 10 + np.random.normal(0, 0.5, len(x))
axs[1, 1].scatter(x, y_sinusoidal, color='orange', alpha=0.7)
axs[1, 1].plot(x, 5 * np.sin(x) + 10, 'r-', linewidth=2)
axs[1, 1].set_title(bidi_text('رابطه سینوسی'), fontsize=21)
axs[1, 1].set_xlabel(bidi_text('ویژگی X'), fontsize=21)
axs[1, 1].set_ylabel(bidi_text('ویژگی Y'), fontsize=21)
axs[1, 1].grid(alpha=0.2)
axs[1, 1].annotate(bidi_text('Y = 5 * sin(X) + 10'), xy=(0.4, 0.9),
                  xycoords='axes fraction', fontsize=21, color='red')

# ایجاد بارکد
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=5, border=2)
qr.add_data("https://github.com/hojatollahgholami/AI-in-Oil-and-gas-industry/edit/main/figs/fig2_12.py")
qr.make(fit=True)

# تبدیل بارکد به تصویر
img = qr.make_image(fill_color="black", back_color="white")
img = img.convert("RGBA")
img_np = np.array(img)

# ایجاد جعبه برای بارکد با تنظیمات صحیح
imagebox = OffsetImage(img_np, zoom=0.6)  
ab = AnnotationBbox(
    imagebox, 
    (0.2, 0.2),  # موقعیت در گوشه پایین سمت راست
    xycoords='figure fraction',  # استفاده از مختصات شکل اصلی
    box_alignment=(1, 0),  # تراز در پایین سمت راست
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به محور فعلی
axs[1, 0].add_artist(ab)

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.25, wspace=0.2)
plt.savefig('fig2-12.png', dpi=300, bbox_inches='tight')
plt.show()
