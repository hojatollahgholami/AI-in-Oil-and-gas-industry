
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import arabic_reshaper
from bidi.algorithm import get_display
import qrcode
from PIL import Image
import matplotlib as mpl

# تابع برای نمایش صحیح متون فارسی
def bidi_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# تنظیمات اولیه
np.random.seed(42)
samples_per_year = 24  # نمونه‌های ماهانه (هر 15 روز)
total_samples = 3 * samples_per_year  # ۳ سال داده

# تولید داده‌های مصنوعی با الگوی فصلی
time = np.arange(total_samples)

# ایجاد الگوی فصلی
seasonal_pattern = np.ones(total_samples)
for year in range(3):
    # تابستان (نمونه‌های 4-8 و 16-20): افزایش 15%
    summer1 = slice(year * samples_per_year + 4, year * samples_per_year + 8)
    summer2 = slice(year * samples_per_year + 16, year * samples_per_year + 20)
    seasonal_pattern[summer1] = 1.15
    seasonal_pattern[summer2] = 1.15

    # زمستان (نمونه‌های 10-14 و 22-2): کاهش 10%
    winter1 = slice(year * samples_per_year + 10, year * samples_per_year + 14)
    winter2 = slice(year * samples_per_year + 22, year * samples_per_year + min(26, (year+1)*samples_per_year))
    seasonal_pattern[winter1] = 0.90
    seasonal_pattern[winter2] = 0.90

# ۱. دو سال اول: میانگین ۲۰۰ کیلووات با نوسان کم
base_consumption = 200 + np.random.normal(0, 3, 2*samples_per_year)  # کاهش بیشتر واریانس

# ۲. سال سوم: روند افزایشی واضح
trend_start = 2 * samples_per_year
trend = 1.8 * (time[trend_start:] - trend_start)  # افزایش بیشتر شیب روند
noise = np.random.normal(0, 3, samples_per_year)
base_third = 200 + trend + noise + np.random.uniform(20, 30)  # افزایش ناگهانی بیشتر

# ترکیب داده‌های پایه
base = np.concatenate([base_consumption, base_third])

# اعمال الگوی فصلی
consumption = base * seasonal_pattern

# ایجاد دیتافریم
df = pd.DataFrame({
    'زمان': time,
    'مصرف_برق': consumption,
    'الگوی_فصلی': seasonal_pattern
})

# محاسبه میانگین‌ها
mean_before = np.mean(consumption[:trend_start])
mean_after = np.mean(consumption[trend_start:])

# محاسبه واریانس متحرک (با پنجره 4 ماهه)
df['واریانس_متحرک'] = df['مصرف_برق'].rolling(window=4).var()

# محاسبه میانگین متحرک برای خط روند
df['میانگین_متحرک'] = df['مصرف_برق'].rolling(window=4).mean()

# تنظیم فونت
plt.rcParams['font.family'] = 'B Nazanin'
plt.rcParams['font.size'] = 18
mpl.rcParams['axes.unicode_minus'] = False  # حل مشکل نمایش علامت منفی

# ایجاد نمودارها
fig = plt.figure(figsize=(15, 14))

# نمودار ۱: سری زمانی مصرف برق
ax1 = plt.subplot(3, 1, 1)

# مشخص کردن مناطق فصلی
for year in range(3):
    # تابستان
    ax1.axvspan(year * samples_per_year + 4, year * samples_per_year + 8,
                alpha=0.2, color='red', label=bidi_text('تابستان') if year == 0 else "")
    ax1.axvspan(year * samples_per_year + 16, year * samples_per_year + 20,
                alpha=0.2, color='red')

    # زمستان
    ax1.axvspan(year * samples_per_year + 10, year * samples_per_year + 14,
                alpha=0.2, color='blue', label=bidi_text('زمستان') if year == 0 else "")
    ax1.axvspan(year * samples_per_year + 22, min((year+1)*samples_per_year, year * samples_per_year + 26),
                alpha=0.2, color='blue')

# رسم داده‌ها و خط روند
ax1.plot(df['زمان'], df['مصرف_برق'], color='royalblue', linewidth=1,
         alpha=0.7, label=bidi_text('داده‌های واقعی'))
ax1.plot(df['زمان'], df['میانگین_متحرک'], color='red', linewidth=2.5,
         label=bidi_text('روند مصرف (میانگین متحرک)'))
ax1.axvline(x=trend_start, color='red', linestyle='--', alpha=0.7)

# خطوط میانگین
ax1.axhline(y=mean_before, color='green', linestyle='-', alpha=0.7,
           label=bidi_text(f'میانگین قبل از تغییر'))
ax1.axhline(y=mean_after, color='purple', linestyle='-', alpha=0.7,
           label=bidi_text(f'میانگین بعد از تغییر'))

ax1.set_title(bidi_text('سری زمانی مصرف برق یک کارخانه'), fontsize=18)
ax1.set_xlabel(bidi_text('زمان (نمونه‌های ۱۵ روزه)'), fontsize=18)
ax1.set_ylabel(bidi_text('مصرف برق (کیلووات)'), fontsize=18)
ax1.set_xlim(0, 70)
ax1.grid(alpha=0.3)
ax1.annotate(bidi_text('افزودن تجهیز جدید'),
             xy=(trend_start, 200),
             xytext=(trend_start-10, 240),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=18)
ax1.legend(loc='upper left', ncol=2)

# نمودار ۲: واریانس متحرک
ax2 = plt.subplot(3, 1, 2)
ax2.plot(df['زمان'], df['واریانس_متحرک'], color='green', linewidth=1.5)
ax2.axvline(x=trend_start, color='red', linestyle='--', alpha=0.7)
ax2.set_title(bidi_text('واریانس متحرک مصرف برق (پنجره ۲ ماهه)'), fontsize=18)
ax2.set_xlabel(bidi_text('زمان (نمونه‌های ۱۵ روزه)'), fontsize=18)
ax2.set_ylabel(bidi_text('واریانس'), fontsize=18)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 900)  # کاهش محدوده محور Y
ax2.set_xlim(0, 70)
# نمودار ۳: خودهمبستگی
ax3 = plt.subplot(3, 1, 3)
plot_acf(df['مصرف_برق'], lags=24, ax=ax3, color='purple')
ax3.set_title(bidi_text('نمودار خودهمبستگی'), fontsize=18)
ax3.set_xlabel(bidi_text('لگ'), fontsize=18)
ax3.set_ylabel(bidi_text('ضریب خودهمبستگی'), fontsize=18)
ax3.grid(alpha=0.3)
ax3.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()

# ایجاد بارکد برای لینک 
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L,box_size=5, border=2)
qr.add_data("https://github.com/hojatollahgholami/AI-in-Oil-and-gas-industry/edit/main/figs/fig2_7.py")
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
                    box_alignment=(8.5, -1.2),
                    frameon=False,
                    pad=0)

# اضافه کردن بارکد به نمودار
ax2.add_artist(ab)

plt.savefig('electricity.png', dpi=300)
plt.show()
