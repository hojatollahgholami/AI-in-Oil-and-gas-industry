
import matplotlib.pyplot as plt
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# تنظیمات فونت فارسی
plt.rcParams['font.family'] = 'Adobe Arabic'
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.size'] = 20

# تابع کمکی برای نمایش صحیح متون فارسی
def fa(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# زمان
time = np.linspace(0, 10, 200)

# سیگنال ارتعاشی واکنشی
vibration_reactive = np.sin(2 * np.pi * 0.5 * time)
vibration_reactive += 0.3 * np.sin(2 * np.pi * 2 * time)
vibration_reactive += np.where((time > 6), 1.5 * np.exp(-(time - 6)), 0)

# سیگنال ارتعاشی پیش‌بینانه
vibration_predictive = np.sin(2 * np.pi * 0.5 * time)
vibration_predictive += 0.3 * np.sin(2 * np.pi * 2 * time)
vibration_predictive += np.where((time > 5), 0.6 * np.exp(-(time - 5)), 0)

threshold = 1.2  # آستانه تریپ

fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# نمودار واکنشی
axs[0].plot(time, vibration_reactive, color='black', label=fa('لرزش یاتاقان (واکنشی)'))
axs[0].axhline(threshold, color='red', linestyle='--', label=fa('آستانه خطر'))
axs[0].fill_between(time, vibration_reactive, threshold,
                    where=(vibration_reactive > threshold), color='orange', alpha=0.5, label=fa('منطقه خطر'))
axs[0].text(7, threshold + 0.2, fa('فرمان واکنشی صادر شد'), color='red', fontsize=20)
axs[0].set_ylabel(fa('مقدار لرزش'))
axs[0].set_title(fa('حالت واکنشی'))
#axs[0].legend()
axs[0].grid(True)

# نوار هزینه بالا
axs[0].barh(2.5, 10, height=0.3, left=0, color='crimson')
axs[0].text(5, 2.4, fa('هزینه تعمیر بالا'), ha='center', va='bottom', color='black', fontweight='bold')

# نمودار پیش‌بینانه
axs[1].plot(time, vibration_predictive, color='black', label=fa('لرزش یاتاقان'))
axs[1].axhline(threshold, color='red', linestyle='--', label=fa('آستانه خطر'))
axs[1].fill_between(time, vibration_predictive, threshold,
                    where=(vibration_predictive > threshold), color='orange', alpha=0.5, label=fa('منطقه هشدار'))
axs[1].text(6, threshold + 0.4, fa('فرمان پیش بینانه صادر شد'), color='green', fontsize=20)
axs[1].set_xlabel(fa('زمان'))
axs[1].set_ylabel(fa('مقدار لرزش'))
axs[1].set_title(fa('حالت پیش بینانه'))
axs[1].legend()
axs[1].grid(True)

# نوار هزینه پایین
axs[1].barh(2.5, 7, height=0.3, left=0, color='seagreen')
axs[1].text(5, 2.35, fa('هزینه تعمیر پایین'), ha='center', va='bottom', color='black', fontweight='bold')

# افزودن فلش و جعبه‌های مراحل بین نمودارها
fig.subplots_adjust(hspace=0.5)
fig.text(0.5, 0.45, '', ha='center')

# فلش از بالا به پایین بین دو نمودار
axs[0].annotate('', xy=(0.49, 0.55), xytext=(0.49, 0.555),
             xycoords='figure fraction', textcoords='figure fraction',
             arrowprops=dict(arrowstyle='->', linewidth=2, color='black'))

# جعبه اول: جمع‌آوری داده‌ها
fig.text(0.5, 0.55, fa('جمع آوری داده ها'), ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black'))

# فلش  بین جمع آوری و پیش پردازش
axs[0].annotate('', xy=(0.49, 0.52), xytext=(0.49, 0.58),
             xycoords='figure fraction', textcoords='figure fraction',
             arrowprops=dict(arrowstyle='->', linewidth=2, color='black'))

# فلش بعد از پیش‌پردازش
axs[0].annotate('', xy=(0.49, 0.485), xytext=(0.49, 0.52),
             xycoords='figure fraction', textcoords='figure fraction',
             arrowprops=dict(arrowstyle='->', linewidth=2, color='black'))

# جعبه دوم: پیش‌پردازش و مدل‌سازی
fig.text(0.5, 0.49, fa('پیش پردازش و مدل سازی'), ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black'))

# ایجاد و تنظیم بارکد
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=3,
    border=2
)
qr.add_data("https://B2n.ir/fig4_2")
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white")

# تبدیل به تصویر قابل استفاده در matplotlib
img = img.convert("RGBA")
img_np = np.array(img)

# تنظیم موقعیت بارکد (پایین سمت راست)
imagebox = OffsetImage(img_np, zoom=0.8)
ab = AnnotationBbox(
    imagebox, 
    (0.12, 0.58),
    xycoords='figure fraction',
    box_alignment=(1, 0),
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به شکل
fig.add_artist(ab)

plt.tight_layout()
plt.savefig('fig4_2.png', dpi=300, bbox_inches='tight')
plt.show()