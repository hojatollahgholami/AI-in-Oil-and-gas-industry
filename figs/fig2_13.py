!pip install qrcode arabic_reshaper python-bidi 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import arabic_reshaper
from bidi.algorithm import get_display
import qrcode
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# تابع برای نمایش صحیح متون فارسی
def fa_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# Load the dataset
df = pd.read_csv('equipment_anomaly_data.csv')

# Identify and separate label and features
label_col = [col for col in df.columns if col.lower() == 'faulty'][0]
y = df[label_col]
X = df.drop(columns=[label_col])

# Remove equipment-related features
X = X.drop(columns=[col for col in X.columns if col.lower() == 'equipment'], errors='ignore')
X = X.loc[:, ~X.columns.str.lower().str.startswith('equipment_')]

# Remove location-related features
X = X.drop(columns=[col for col in X.columns if col.lower() == 'location'], errors='ignore')
X = X.loc[:, ~X.columns.str.lower().str.startswith('location_')]

# Encode remaining string columns
string_cols = X.select_dtypes(include=['object']).columns.tolist()
if string_cols:
    X = pd.get_dummies(X, columns=string_cols, drop_first=True)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
scores = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_

# ایجاد نمودارها
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
plt.rcParams['font.family'] = 'Adobe Arabic'
plt.rcParams['font.size'] = 21

# Scree plot with cumulative variance and 0.9 line
cumulative_variance = explained_variance.cumsum()
num_components = len(explained_variance)

ax1.bar(range(1, num_components + 1), explained_variance, alpha=0.6, label=fa_text('واریانس هر مؤلفه'))
ax1.plot(range(1, num_components + 1), cumulative_variance, marker='o', color='red', label=fa_text('واریانس تجمعی'))

# خط افقی روی ۹۰٪
ax1.axhline(y=0.9, color='gray', linestyle='--', linewidth=1)
ax1.text(1, 0.91, '0.9', color='gray')

ax1.set_xlabel(fa_text('مولفه اصلی'))
ax1.set_ylabel(fa_text('واریانس'))
ax1.set_title(fa_text('نمودار اسکری با واریانس تجمعی'))
ax1.set_xticks(range(1, num_components + 1))
ax1.set_ylim(0, 1.05)
ax1.legend()

#biplot 
colors = ['green' if lbl == 0 else 'red' for lbl in y]
ax2.scatter(scores[:, 0], scores[:, 1], c=colors, alpha=0.6, edgecolors='k', linewidth=0.2)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_title(fa_text('Biplot نمودار'))
for i, feature in enumerate(X.columns):
    ax2.arrow(0, 0,
              pca.components_[0, i] * max(scores[:, 0]),
              pca.components_[1, i] * max(scores[:, 1]),
              head_width=0.03 * max(scores[:, 0]),
              head_length=0.03 * max(scores[:, 1]),
              linewidth=1)
    ax2.text(pca.components_[0, i] * max(scores[:, 0]) * 1.1,
             pca.components_[1, i] * max(scores[:, 1]) * 1.1,
             feature, fontsize=24)

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
    (0.14, 0.55),  
    xycoords='figure fraction',  # استفاده از مختصات شکل اصلی
    box_alignment=(1, 0), 
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به محور فعلی
ax1.add_artist(ab)

# Remove axes, ticks, and title
plt.tight_layout()
plt.savefig('fig2-13.png', dpi=300, bbox_inches='tight')
plt.show()
