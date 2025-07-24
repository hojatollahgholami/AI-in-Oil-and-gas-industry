import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pywt
from scipy.optimize import curve_fit
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import qrcode
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# تابع برای نمایش صحیح متون فارسی
def bidi_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# تنظیمات اولیه
plt.rcParams['font.family'] = 'Adobe Arabic'
plt.rcParams['font.size'] = 21
mpl.rcParams['axes.unicode_minus'] = False

# تولید داده‌های مصنوعی برای صنعت نفت و گاز (دمای فرآیند)
np.random.seed(42)
t = np.linspace(0, 24, 1000)  # زمان بر حسب ساعت
freq1 = 0.5  # فرکانس روزانه
freq2 = 2.0  # فرکانس نوسانات سریع

# سیگنال اصلی: ترکیب سینوسی + نویز + پالس گوسی
signal = (
    80 + 10 * np.sin(2 * np.pi * freq1 * t) +  # سیگنال روزانه
    3 * np.sin(2 * np.pi * freq2 * t) +       # نوسانات سریع
    np.random.normal(0, 1.5, len(t))          # نویز تصادفی
)

# افزودن پالس گوسی (شبیه‌سازی رویداد غیرعادی)
gaussian_pulse = 25 * np.exp(-0.5 * ((t - 12) / 0.5)**2)
signal += gaussian_pulse

# ایجاد نمودار اصلی با ساختار GridSpec
fig = plt.figure(figsize=(20, 18))

gs = GridSpec(4, 2, figure=fig, height_ratios=[1.5, 1, 1, 1])

# 1. نمودار سیگنال اصلی و گوسی
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, signal, 'b-', linewidth=1.5, alpha=0.8, label=bidi_text('سیگنال اصلی'))
ax1.plot(t, gaussian_pulse, 'r--', linewidth=2, label=bidi_text('پالس گوسی (رویداد غیرعادی)'))
ax1.set_title(bidi_text('سیگنال دمای فرآیند در صنعت نفت'), fontsize=24)
ax1.set_xlabel(bidi_text('زمان (ساعت)'), fontsize=24)
ax1.set_ylabel(bidi_text('دما (°C)'), fontsize=24)
ax1.grid(alpha=0.2)
ax1.legend(fontsize=21)
ax1.set_xlim(0, 24)
ax1.axvline(12, color='g', linestyle='--', alpha=0.5, label=bidi_text('زمان رویداد (12 ساعت)'))
ax1.legend(loc='right')

# 2. تبدیل فوریه
N = len(t)
T = t[1] - t[0]  # فاصله زمانی
yf = fft(signal - np.mean(signal))
xf = fftfreq(N, T)[:N//2]

# محاسبه طیف توان
power_spectrum = 2.0/N * np.abs(yf[0:N//2])

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(xf, power_spectrum, 'g-', linewidth=2)
ax2.set_title(bidi_text('تبدیل فوریه: طیف فرکانسی'), fontsize=24)
ax2.set_xlabel(bidi_text('فرکانس (Hz)'), fontsize=24)
ax2.set_ylabel(bidi_text('توان'), fontsize=24)
ax2.grid(alpha=0.2)
ax2.set_xlim(0, 3)

# مشخص کردن پیک‌های اصلی
peaks = [0.5, 2.0]
for peak in peaks:
    ax2.axvline(peak, color='r', linestyle='--', alpha=0.7)
    ax2.text(peak + 0.1, max(power_spectrum)*0.9,
            bidi_text(f'فرکانس: {peak} Hz'),
            color='red', fontsize=21)

# 3. تبدیل موجک (CWT)
scales = np.arange(1, 128)
wavelet = 'cmor1.5-1.0'  # موجک مورلت مختلط
coef, freqs = pywt.cwt(signal - np.mean(signal), scales, wavelet, 1/T)

ax3 = fig.add_subplot(gs[1, 1])
cwt_plot = ax3.contourf(t, scales, np.abs(coef), cmap='viridis', levels=100)
ax3.set_title(bidi_text('تبدیل موجک: آنالیز زمان-فرکانس'), fontsize=24)
ax3.set_xlabel(bidi_text('زمان (ساعت)'), fontsize=24)
ax3.set_ylabel(bidi_text('مقیاس'), fontsize=24)
ax3.invert_yaxis()
fig.colorbar(cwt_plot, ax=ax3, label=bidi_text('دامنه'))

# مشخص کردن رویداد غیرعادی
ax3.axvline(12, color='r', linestyle='--', alpha=0.8)
ax3.text(12.2, 70, bidi_text('رویداد غیرعادی'), color='red', fontsize=21, rotation=90)

# 4. معادله گوسی برای رویداد غیرعادی
def gaussian(x, a, b, c, d):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

# انتخاب داده‌های اطراف رویداد
t_event = t[(t >= 10) & (t <= 14)]
signal_event = signal[(t >= 10) & (t <= 14)]

# برازش منحنی گوسی
p0 = [25, 12, 0.5, 80]  # حدس اولیه: دامنه، میانگین، انحراف معیار، آفست
popt, pcov = curve_fit(gaussian, t_event, signal_event, p0=p0)
a, b, c, d = popt

# محاسبه منحنی برازش شده
fit_curve = gaussian(t_event, a, b, c, d)

ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(t_event, signal_event, 'b-', linewidth=2, label=bidi_text('سیگنال اصلی'))
ax4.plot(t_event, fit_curve, 'r--', linewidth=3, label=bidi_text('برازش گوسی'))
ax4.set_title(bidi_text('برازش منحنی گوسی برای رویداد غیرعادی'), fontsize=24)
ax4.set_xlabel(bidi_text('زمان (ساعت)'), fontsize=24)
ax4.set_ylabel(bidi_text('دما (°C)'), fontsize=24)
ax4.grid(alpha=0.2)
ax4.legend(fontsize=21)

# نمایش معادله گوسی (نسخه اصلاح شده)
equation_text = bidi_text(f'معادله: f(t) = {a:.1f} * exp(-(t-{b:.1f})²/(2*{c:.2f}²)) + {d:.1f}')
ax4.text(0.5, 0.9, equation_text, transform=ax4.transAxes,
        fontsize=14, color='red', ha='center',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# 5. تحلیل مولفه‌های اصلی با تبدیل فوریه
# جداسازی سیگنال به سه بخش: پایه، نوسانات سریع، پالس گوسی
base_signal = 80 + 10 * np.sin(2 * np.pi * freq1 * t)
fast_osc = 3 * np.sin(2 * np.pi * freq2 * t)
noise = np.random.normal(0, 1.5, len(t))

ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(t, base_signal, 'g-', linewidth=2, label=bidi_text('سیگنال پایه (روزانه)'))
ax5.plot(t, fast_osc, 'b-', linewidth=1, alpha=0.7, label=bidi_text('نوسانات سریع'))
ax5.plot(t, noise, 'm-', linewidth=0.5, alpha=0.5, label=bidi_text('نویز تصادفی'))
ax5.plot(t, gaussian_pulse, 'r-', linewidth=2, label=bidi_text('پالس گوسی'))
ax5.set_title(bidi_text('مولفه‌های تشکیل‌دهنده سیگنال'), fontsize=24)
ax5.set_xlabel(bidi_text('زمان (ساعت)'), fontsize=24)
ax5.set_ylabel(bidi_text('دما (°C)'), fontsize=24)
ax5.grid(alpha=0.2)
ax5.legend(fontsize=21)
ax5.set_xlim(0, 24)

# 6. مقایسه روش‌ها در تشخیص رویداد
ax6 = fig.add_subplot(gs[3, :])
methods = [
    bidi_text('سیگنال اصلی'),
    bidi_text('تبدیل فوریه'),
    bidi_text('تبدیل موجک'),
    bidi_text('برازش گوسی')
]
scores = [0.6, 0.3, 0.9, 0.95]  # امتیاز تشخیص رویداد

bars = ax6.bar(methods, scores, color=['blue', 'green', 'purple', 'red'])
ax6.set_title(bidi_text('مقایسه روش‌ها در تشخیص رویداد غیرعادی'), fontsize=24)
ax6.set_ylabel(bidi_text('دقت تشخیص'), fontsize=24)
ax6.grid(axis='y', alpha=0.2)
ax6.set_ylim(0, 1.1)

# افزودن مقادیر روی میله‌ها
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.2f}',
            ha='center', fontsize=12)

# ایجاد بارکد
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=5, border=2)
qr.add_data("https://B2n.ir/hr4292")
qr.make(fit=True)

# تبدیل بارکد به تصویر
img = qr.make_image(fill_color="black", back_color="white")
img = img.convert("RGBA")
img_np = np.array(img)

# ایجاد جعبه برای بارکد با تنظیمات صحیح
imagebox = OffsetImage(img_np, zoom=0.8)  
ab = AnnotationBbox(
    imagebox, 
    (0.2, 0.75),  
    xycoords='figure fraction',  # استفاده از مختصات شکل اصلی
    box_alignment=(1, 0), 
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به محور فعلی
ax1.add_artist(ab)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.2)
plt.savefig('fig2_17.png', dpi=300, bbox_inches='tight')
plt.show()
