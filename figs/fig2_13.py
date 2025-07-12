!pip install qrcode[pil] arabic_reshaper python-bidi 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
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
plt.rcParams['font.family'] = 'Adobe Arabic'
plt.rcParams['font.size'] = 21
mpl.rcParams['axes.unicode_minus'] = False

# ایجاد داده‌های مصنوعی برای صنعت نفت و گاز
np.random.seed(42)
X, y = make_classification(
    n_samples=300,
    n_features=8,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    class_sep=1.5,
    random_state=42
)

# نام‌های ویژگی‌ها به فارسی
feature_names = [
    bidi_text('دما'),
    bidi_text('فشار'),
    bidi_text('لرزش'),
    bidi_text('جریان'),
    bidi_text('رطوبت'),
    bidi_text('غلظت'),
    bidi_text('اسیدیته'),
    bidi_text('رسانایی')
]

# اعمال PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# محاسبه واریانس تجمعی
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# ایجاد نمودارها
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# 1. نمودار پراکنش دو مؤلفه اصلی
scatter = ax1.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=y, cmap='viridis',
    alpha=0.8, edgecolor='w', s=80
)
ax1.set_title(bidi_text('نمودار پراکنش دو مؤلفه اصلی'), fontsize=24)
ax1.set_xlabel(bidi_text('مؤلفه اصلی ۱ (PC1)'), fontsize=24)
ax1.set_ylabel(bidi_text('مؤلفه اصلی ۲ (PC2)'), fontsize=24)
ax1.grid(alpha=0.2)

# افزودن بردارهای ویژه (بارهای مؤلفه‌ها)
for i, (x, y_dir) in enumerate(zip(pca.components_[0], pca.components_[1])):
    ax1.arrow(0, 0, x*3, y_dir*3, color='r', width=0.02, head_width=0.1)
    ax1.text(x*3.2, y_dir*3.2, feature_names[i], color='red', fontsize=21)

# افزودن توضیحات کلاس‌ها
legend = ax1.legend(*scatter.legend_elements(),
                   title=bidi_text('کلاس‌ها'),
                   loc='upper right')
ax1.add_artist(legend)

# 2. نمودار Scree (واریانس توضیح داده شده)
ax2.bar(
    range(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_,
    color='skyblue',
    alpha=0.8,
    label=bidi_text('واریانس هر مؤلفه')
)

# خط واریانس تجمعی
ax2.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    'ro-',
    linewidth=2,
    markersize=8,
    label=bidi_text('واریانس تجمعی')
)

ax2.set_title(bidi_text('نمودار اسکری واریانس توضیح داده شده توسط مؤلفه‌ها'), fontsize=24)
ax2.set_xlabel(bidi_text('شماره مؤلفه اصلی'), fontsize=24)
ax2.set_ylabel(bidi_text('درصد واریانس توضیح داده شده'), fontsize=24)
ax2.set_xticks(range(1, len(pca.explained_variance_ratio_) + 1))
ax2.grid(axis='y', alpha=0.2)
ax2.legend(fontsize=21)

# افزودن مقادیر روی نمودار Scree
for i, (v, c) in enumerate(zip(pca.explained_variance_ratio_, cumulative_variance)):
    ax2.text(i+1, v+0.01, f'{v*100:.1f}%', ha='center', fontsize=18)
    ax2.text(i+1, c+0.01, f'{c*100:.1f}%', ha='center', fontsize=18)

# افزودن خط آستانه 90%
ax2.axhline(y=0.9, color='g', linestyle='--', alpha=0.7)
ax2.text(1, 0.91, bidi_text('آستانه ۹۰٪'), color='g', fontsize=21)

# ایجاد بارکد
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=5, border=2)
qr.add_data("https://B2n.ir/xb5100")
qr.make(fit=True)

# تبدیل بارکد به تصویر
img = qr.make_image(fill_color="black", back_color="white")
img = img.convert("RGBA")
img_np = np.array(img)

# ایجاد جعبه برای بارکد با تنظیمات صحیح
imagebox = OffsetImage(img_np, zoom=0.8)  
ab = AnnotationBbox(
    imagebox, 
    (0.14, 0.1),  
    xycoords='figure fraction',  # استفاده از مختصات شکل اصلی
    box_alignment=(1, 0), 
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به محور فعلی
ax1.add_artist(ab)

plt.tight_layout()
plt.savefig('fig2-13.png', dpi=300, bbox_inches='tight')
plt.show()
