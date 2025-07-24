import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              StackingClassifier, VotingClassifier,
                              GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display
import qrcode
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# تنظیمات فارسی‌نویسی
plt.rcParams["font.family"] = 'Adobe Arabic'
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.size'] = 31

def persian_text(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

# ایجاد مجموعه داده
X, y = make_circles(n_samples=500, noise=0.25, factor=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# تنظیمات نمودار
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
h = 0.02  # گام شبکه

# مدل‌های پایه برای استکینگ و رأی‌گیری
base_models = [
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svm', SVC(C=1.0, kernel='rbf', probability=True, random_state=42))
]

# ایجاد مدل‌های انسمبل
models = {
    persian_text("درخت تصمیم (پایه)"): DecisionTreeClassifier(max_depth=5, random_state=42),
    persian_text("بگینگ (جنگل تصادفی)"): RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
    persian_text("بوستینگ (AdaBoost)"): AdaBoostClassifier(n_estimators=50, learning_rate=0.8, random_state=42),
    persian_text("گرادیان بوستینگ"): GradientBoostingClassifier(n_estimators=50, learning_rate=0.8, max_depth=3, random_state=42),
    persian_text("استکینگ"): StackingClassifier(estimators=base_models, final_estimator=SVC(), cv=5),
    persian_text("رأی‌گیری (سخت)"): VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('ada', AdaBoostClassifier(n_estimators=10, random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ], voting='hard')
}

# محاسبه محدوده نمودار
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# ایجاد نمودار در ۲ ردیف و ۳ ستون
fig, axs = plt.subplots(2, 3, figsize=(20, 13))

# لیست محورها به صورت یک بعدی
axs_flat = axs.flatten()

for i, (name, model) in enumerate(models.items()):
    ax = axs_flat[i]

    # آموزش مدل
    model.fit(X_train, y_train)

    # محاسبه دقت
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    # پیش‌بینی شبکه
    if hasattr(model, "predict_proba"):
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    # رسم مرز تصمیم
    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.7)

    # رسم داده‌های آموزش و آزمون
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k', alpha=0.6, label=persian_text("آموزش"))
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.9, marker='s', label=persian_text("آزمون"))

    # تنظیمات عنوان و توضیحات
    ax.set_title(f"{name}\n{persian_text('دقت آموزش')}: {train_acc:.2f}, {persian_text('دقت آزمون')}: {test_acc:.2f}", fontsize=24)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend(loc='best')

# ایجاد بارکد
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=5, border=2)
qr.add_data("https://B2n.ir/mm4460")
qr.make(fit=True)

# تبدیل بارکد به تصویر
img = qr.make_image(fill_color="black", back_color="white")
img = img.convert("RGBA")
img_np = np.array(img)

# ایجاد جعبه برای بارکد با تنظیمات صحیح
imagebox = OffsetImage(img_np, zoom=0.8)  
ab = AnnotationBbox(
    imagebox, 
    (0.37, 0.028),  
    xycoords='figure fraction',  # استفاده از مختصات شکل اصلی
    box_alignment=(1, 0), 
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به محور فعلی
ax.add_artist(ab)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('fig2_19.png', dpi=300, bbox_inches='tight')
plt.show()
