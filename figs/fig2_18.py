import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display
import qrcode
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# تنظیمات فارسی‌نویسی
plt.rcParams["font.family"] = "Adobe Arabic"
plt.rcParams["axes.unicode_minus"] = False

def persian_text(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

# تولید داده‌های مصنوعی
np.random.seed(42)
X = np.linspace(0, 2, 100)
y = 0.5 * np.sin(2 * np.pi * X) + 0.3 * np.random.randn(100)

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# مدل‌های مختلف
degrees = [1, 4, 15]
models = {
    "خطی": make_pipeline(PolynomialFeatures(1), LinearRegression()),
    "چندجمله ای (درجه ۴)": make_pipeline(PolynomialFeatures(4), LinearRegression()),
    "چندجمله ای (درجه ۱۵)": make_pipeline(PolynomialFeatures(15), LinearRegression()),
    "تنظیم ریدج (alpha=0.1)": make_pipeline(PolynomialFeatures(15), Ridge(alpha=0.1)),
    "تنظیم لاسو (alpha=0.01)": make_pipeline(PolynomialFeatures(15), Lasso(alpha=0.01))
}

# آموزش مدل‌ها و ذخیره نتایج
results = {}
X_plot = np.linspace(0, 2, 500).reshape(-1, 1)

for name, model in models.items():
    model.fit(X_train.reshape(-1, 1), y_train)
    y_plot = model.predict(X_plot)
    y_pred = model.predict(X_test.reshape(-1, 1))
    mse = mean_squared_error(y_test, y_pred)
    results[name] = (y_plot, mse)

# ترسیم نمودارها
fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

# نمودار ۱: بیش برازش
axs[0].scatter(X_train, y_train, s=20, label=persian_text("داده های آموزش"), alpha=0.6)
axs[0].scatter(X_test, y_test, s=30, marker='x', label=persian_text("داده های آزمون"))
axs[0].plot(X_plot, 0.5 * np.sin(2 * np.pi * X_plot), 'k--', label=persian_text("تابع واقعی"))

for name in list(models.keys())[:3]:
    y_plot, mse = results[name]
    axs[0].plot(X_plot, y_plot, label=f"{persian_text(name)}")

axs[0].set_title(persian_text("نمایش بیش برازش در مدل های چندجمله ای"), fontsize=24)
axs[0].set_xlabel(persian_text("ویژگی (X)"), fontsize=24)
axs[0].set_ylabel(persian_text("هدف (y)"), fontsize=24)
axs[0].legend(loc='upper right', prop={'size': 21})
axs[0].grid(alpha=0.2)

# نمودار ۲: تنظیم مجدد
axs[1].scatter(X_train, y_train, s=20, alpha=0.6)
axs[1].scatter(X_test, y_test, s=30, marker='x')
axs[1].plot(X_plot, 0.5 * np.sin(2 * np.pi * X_plot), 'k--')

for name in list(models.keys())[3:]:
    y_plot, mse = results[name]
    axs[1].plot(X_plot, y_plot, label=f"{persian_text(name)}")

axs[1].set_title(persian_text("اثر تنظیم مجدد در کنترل بیش برازش"), fontsize=24)
axs[1].set_xlabel(persian_text("ویژگی (X)"), fontsize=24)
axs[1].legend(loc='upper right', prop={'size': 21})
axs[1].grid(alpha=0.2)

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
    (0.15, 0.15),  
    xycoords='figure fraction',  # استفاده از مختصات شکل اصلی
    box_alignment=(1, 0), 
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به محور فعلی
axs[0].add_artist(ab)

plt.tight_layout()
plt.savefig('fig2_18.png', dpi=300, bbox_inches='tight')
plt.show()
