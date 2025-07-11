!pip install qrcode[pil] arabic_reshaper python-bidi

import numpy as np
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from scipy.stats import norm, expon, uniform, gamma, binom, poisson
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
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# تنظیمات مشترک برای همه نمودارها
x = np.linspace(-5, 15, 1000)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 1. توزیع نرمال
mu, sigma = 0, 1
axs[0, 0].plot(x, norm.pdf(x, mu, sigma), color=colors[0], linewidth=2)
axs[0, 0].fill_between(x, norm.pdf(x, mu, sigma), color=colors[0], alpha=0.3)
axs[0, 0].set_title(bidi_text('توزیع نرمال (mu=0, sigma=1)'), fontsize=21)
axs[0, 0].set_xlabel(bidi_text('مقدار'), fontsize=21)
axs[0, 0].set_ylabel(bidi_text('چگالی احتمال'), fontsize=21)
axs[0, 0].grid(alpha=0.2)

# 2. توزیع نمایی
lambda_param = 0.5
axs[0, 1].plot(x, expon.pdf(x, scale=1/lambda_param), color=colors[1], linewidth=2)
axs[0, 1].fill_between(x, expon.pdf(x, scale=1/lambda_param), color=colors[1], alpha=0.3)
axs[0, 1].set_title(bidi_text('توزیع نمایی (lambda=0.5)'), fontsize=21)
axs[0, 1].set_xlabel(bidi_text('مقدار'), fontsize=21)
axs[0, 1].set_ylabel(bidi_text('چگالی احتمال'), fontsize=21)
axs[0, 1].grid(alpha=0.2)

# 3. توزیع یکنواخت
a, b = 0, 10
axs[0, 2].plot(x, uniform.pdf(x, a, b-a), color=colors[2], linewidth=2)
axs[0, 2].fill_between(x, uniform.pdf(x, a, b-a), color=colors[2], alpha=0.3)
axs[0, 2].set_title(bidi_text('توزیع یکنواخت (a=0, b=10)'), fontsize=21)
axs[0, 2].set_xlabel(bidi_text('مقدار'), fontsize=21)
axs[0, 2].set_ylabel(bidi_text('چگالی احتمال'), fontsize=21)
axs[0, 2].grid(alpha=0.2)

# 4. توزیع گاما
alpha, beta = 2, 1.5
axs[1, 0].plot(x, gamma.pdf(x, alpha, scale=1/beta), color=colors[3], linewidth=2)
axs[1, 0].fill_between(x, gamma.pdf(x, alpha, scale=1/beta), color=colors[3], alpha=0.3)
axs[1, 0].set_title(bidi_text('توزیع گاما (alpha=2, beta=1.5)'), fontsize=21)
axs[1, 0].set_xlabel(bidi_text('مقدار'), fontsize=21)
axs[1, 0].set_ylabel(bidi_text('چگالی احتمال'), fontsize=21)
axs[1, 0].grid(alpha=0.2)

# 5. توزیع دوجمله‌ای
n, p = 15, 0.4
x_binom = np.arange(0, n+1)
axs[1, 1].bar(x_binom, binom.pmf(x_binom, n, p), color=colors[4], alpha=0.7)
axs[1, 1].set_title(bidi_text('توزیع دوجمله‌ای (n=15, p=0.4)'), fontsize=21)
axs[1, 1].set_xlabel(bidi_text('تعداد موفقیت‌ها'), fontsize=21)
axs[1, 1].set_ylabel(bidi_text('احتمال'), fontsize=21)
axs[1, 1].grid(alpha=0.2)

# 6. توزیع پواسون
mu_poisson = 4
x_poisson = np.arange(0, 15)
axs[1, 2].bar(x_poisson, poisson.pmf(x_poisson, mu_poisson), color=colors[5], alpha=0.7)
axs[1, 2].set_title(bidi_text('توزیع پواسون (lambda=4)'), fontsize=21)
axs[1, 2].set_xlabel(bidi_text('رخدادها'), fontsize=21)
axs[1, 2].set_ylabel(bidi_text('احتمال'), fontsize=21)
axs[1, 2].grid(alpha=0.2)

# ایجاد بارکد
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=5, border=2)
qr.add_data("https://github.com/hojatollahgholami/AI-in-Oil-and-gas-industry/edit/main/figs/fig2_11.py")
qr.make(fit=True)

# تبدیل بارکد به تصویر
img = qr.make_image(fill_color="black", back_color="white")
img = img.convert("RGBA")
img_np = np.array(img)

# ایجاد جعبه برای بارکد با تنظیمات صحیح
imagebox = OffsetImage(img_np, zoom=0.6)  
ab = AnnotationBbox(
    imagebox, 
    (0.3, 0.68),  # موقعیت در گوشه پایین سمت راست
    xycoords='figure fraction',  # استفاده از مختصات شکل اصلی
    box_alignment=(1, 0),  # تراز در پایین سمت راست
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به محور فعلی
axs[0, 0].add_artist(ab)

plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3, wspace=0.2)
plt.savefig('fig2-11.png', dpi=300, bbox_inches='tight')
plt.show()
