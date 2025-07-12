!pip install qrcode[pil] arabic_reshaper python-bidi factor_analyzer

import numpy as np
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from sklearn.datasets import make_spd_matrix
import arabic_reshaper
from bidi.algorithm import get_display
import seaborn as sns
import matplotlib as mpl
import qrcode
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# تابع برای نمایش صحیح متون فارسی
def bidi_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# تنظیمات اولیه
plt.rcParams['font.family'] = 'B Nazanin'
plt.rcParams['font.size'] = 21
mpl.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

# ایجاد داده‌های مصنوعی برای صنعت نفت و گاز
n_samples = 300
n_features = 10

# نام‌های متغیرها به فارسی
feature_names = [
    bidi_text('دمای راکتور'),
    bidi_text('فشار مخزن'),
    bidi_text('ارتعاش پمپ'),
    bidi_text('جریان خروجی'),
    bidi_text('غلظت کاتالیست'),
    bidi_text('اسیدیته محصول'),
    bidi_text('رسانایی سیال'),
    bidi_text('ناخالصی‌ها'),
    bidi_text('رطوبت گاز'),
    bidi_text('آلودگی محیطی')
]

# ایجاد ماتریس همبستگی ساختگی
corr_matrix = make_spd_matrix(n_features)

# افزودن ساختار عاملی
# عامل 1: متغیرهای فرآیندی (0-3)
# عامل 2: متغیرهای کیفیتی (4-7)
# عامل 3: متغیرهای محیطی (8-9)
corr_matrix[:4, :4] = corr_matrix[:4, :4] * 0.3 + 0.7
corr_matrix[4:8, 4:8] = corr_matrix[4:8, 4:8] * 0.4 + 0.6
corr_matrix[8:, 8:] = corr_matrix[8:, 8:] * 0.2 + 0.8

# ایجاد داده‌ها بر اساس ماتریس همبستگی
mean = np.zeros(n_features)
X = np.random.multivariate_normal(mean, corr_matrix, size=n_samples)

# تحلیل عاملی اکتشافی
efa = FactorAnalyzer(n_factors=3, rotation='varimax')
efa.fit(X)

# دریافت بارهای عاملی
loadings = efa.loadings_

# ایجاد نمودارها
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

# 1. نمودار حرارتی بارهای عاملی
sns.heatmap(
    loadings,
    annot=True, fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    ax=ax1,
    cbar_kws={'label': bidi_text('بار عاملی')},
    annot_kws={'fontsize': 18}
)

ax1.set_title(bidi_text('بارهای عاملی'), fontsize=24)
ax1.set_xlabel(bidi_text('عوامل'), fontsize=24)
ax1.set_ylabel(bidi_text('متغیرها'), fontsize=24)
ax1.set_xticklabels([
    bidi_text('عامل ۱: فرآیندی'),
    bidi_text('عامل ۲: کیفیتی'),
    bidi_text('عامل ۳: محیطی')
], fontsize=21)
ax1.set_yticklabels(feature_names, rotation=360, fontsize=21)

# 2. نمودار بارهای عاملی برای هر عامل
factors = [bidi_text('عامل ۱: فرآیندی'), bidi_text('عامل ۲: کیفیتی'), bidi_text('عامل ۳: محیطی')]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, factor in enumerate(factors):
    # مرتب‌سازی بارهای عاملی
    sorted_idx = np.argsort(loadings[:, i])
    sorted_loadings = loadings[sorted_idx, i]
    sorted_names = [feature_names[j] for j in sorted_idx]

    # ایجاد نمودار میله‌ای
    bars = ax2.barh(
        np.arange(len(sorted_names)),
        sorted_loadings,
        color=colors[i],
        alpha=0.7,
        label=factor
    )

    # افزودن مقادیر بارها
    for j, bar in enumerate(bars):
        value = bar.get_width()
        ax2.text(
            value + 0.05 * np.sign(value),
            j,
            f'{value:.2f}',
            va='center',
            fontsize=18,
            color=colors[i]
        )

ax2.set_title(bidi_text('بارهای عاملی به تفکیک متغیرها'), fontsize=24)
ax2.set_xlabel(bidi_text('بار عاملی'), fontsize=24)
ax2.set_ylabel(bidi_text('متغیرها'), fontsize=24)
ax2.set_yticks(np.arange(len(sorted_names)))
ax2.set_yticklabels(sorted_names, fontsize=21)
ax2.axvline(0, color='gray', linestyle='--', alpha=0.7)
ax2.axvline(0.4, color='red', linestyle=':', alpha=0.5)
ax2.axvline(-0.4, color='red', linestyle=':', alpha=0.5)
ax2.grid(axis='x', alpha=0.2)
ax2.legend(fontsize=21, loc='lower right')

# ایجاد بارکد
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=5, border=2)
qr.add_data("https://B2n.ir/jw8702")
qr.make(fit=True)

# تبدیل بارکد به تصویر
img = qr.make_image(fill_color="black", back_color="white")
img = img.convert("RGBA")
img_np = np.array(img)

# ایجاد جعبه برای بارکد با تنظیمات صحیح
imagebox = OffsetImage(img_np, zoom=0.8)  
ab = AnnotationBbox(
    imagebox, 
    (0.65, 0.63),  
    xycoords='figure fraction',  # استفاده از مختصات شکل اصلی
    box_alignment=(1, 0), 
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به محور فعلی
ax2.add_artist(ab)

plt.tight_layout()
plt.subplots_adjust(top=0.90, wspace=0.25)
plt.savefig('fig2_14.png', dpi=300, bbox_inches='tight')
plt.show()
