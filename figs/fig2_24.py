import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display
import qrcode
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# تنظیمات فارسی‌نویسی
plt.rcParams["font.family"] = "Adobe Arabic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.size'] = 22

def persian_text(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

# =============================================================================
# تولید داده‌های مصنوعی برای طبقه‌بندی دودویی
# =============================================================================
X, y = make_classification(
    n_samples=1000,  # تعداد نمونه‌ها
    n_features=20,   # تعداد ویژگی‌ها
    n_informative=8, # ویژگی‌های اطلاعاتی
    n_redundant=4,   # ویژگی‌های تکراری
    n_classes=2,     # تعداد کلاس‌ها
    random_state=42
)

# تقسیم داده به آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =============================================================================
# آموزش مدل رگرسیون لجستیک
# =============================================================================
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# پیش‌بینی احتمالات برای داده آزمون
y_proba = model.predict_proba(X_test)[:, 1]  # احتمالات کلاس مثبت

# =============================================================================
# محاسبه معیارهای ROC و AUC
# =============================================================================
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# =============================================================================
# ترسیم منحنی ROC
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# رسم منحنی ROC
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'{persian_text("منحنی ROC")} (AUC = {roc_auc:.3f})')

# خط مبنا (تصادفی)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label=persian_text('مدل تصادفی'))

# تنظیمات نمودار
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel(persian_text('نرخ مثبت کاذب (FPR)'), fontsize=21)
plt.ylabel(persian_text('نرخ مثبت واقعی (TPR)'), fontsize=21)
#plt.title(persian_text('منحنی مشخصه عملکرد گیرنده (ROC)'), fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)

# هایلایت نقطه بهینه (نزدیک به گوشه چپ بالا)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10)
plt.annotate(
    persian_text(f'نقطه بهینه (آستانه = {optimal_threshold:.2f})'),
    xy=(fpr[optimal_idx], tpr[optimal_idx]),
    xytext=(0.4, 0.3),
    arrowprops=dict(facecolor='red', arrowstyle='->'),
    fontsize=22
)

# افزودن اطلاعات عملکرد مدل
plt.text(0.6, 0.22, persian_text(f'مساحت زیر منحنی AUC = {roc_auc:.3f}'),
         fontsize=22, bbox=dict(facecolor='white', alpha=0.8))
plt.text(0.6, 0.15, persian_text(f'نقطه بهینه (TPR={tpr[optimal_idx]:.2f}, FPR={fpr[optimal_idx]:.2f})'),
         fontsize=22, bbox=dict(facecolor='white', alpha=0.8))

# ایجاد بارکد
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=5, border=2)
qr.add_data("https://B2n.ir/rs9366")
qr.make(fit=True)

# تبدیل بارکد به تصویر
img = qr.make_image(fill_color="black", back_color="white")
img = img.convert("RGBA")
img_np = np.array(img)

# ایجاد جعبه برای بارکد با تنظیمات صحیح
imagebox = OffsetImage(img_np, zoom=0.8)  
ab = AnnotationBbox(
    imagebox, 
    (0.3, 0.7),  
    xycoords='figure fraction',  # استفاده از مختصات شکل اصلی
    box_alignment=(1, 0), 
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به محور فعلی
ax.add_artist(ab)

# ذخیره و نمایش نمودار
plt.tight_layout()
plt.savefig('fig2_24.png', dpi=300)
plt.show()
