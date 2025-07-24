import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# تولید داده‌های مصنوعی برای صنعت نفت و گاز
np.random.seed(42)

# 1. داده‌های رگرسیون خطی (رابطه بین فشار و دمای مخزن)
pressure = np.linspace(50, 200, 100)
temperature = 0.8 * pressure + 30 + np.random.normal(0, 10, len(pressure))

# مدل رگرسیون خطی
linear_model = LinearRegression()
linear_model.fit(pressure.reshape(-1, 1), temperature)
linear_pred = linear_model.predict(pressure.reshape(-1, 1))

# 2. داده‌های رگرسیون غیرخطی (رابطه بین زمان کارکرد و سایش تجهیز)
time = np.linspace(0, 24, 100)
wear = 0.05 * time**2 + 0.8 * time + 10 + np.random.normal(0, 3, len(time))

# مدل رگرسیون غیرخطی (پلی‌نومیال درجه 2)
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(time.reshape(-1, 1), wear)
poly_pred = poly_model.predict(time.reshape(-1, 1))

# 1. نمودار رگرسیون خطی
ax1.scatter(pressure, temperature, color='blue', alpha=0.7, label=bidi_text('داده‌های واقعی'))
ax1.plot(pressure, linear_pred, 'r-', linewidth=3, label=bidi_text('رگرسیون خطی'))
ax1.set_title(bidi_text('رگرسیون خطی: رابطه فشار و دما در مخزن'), fontsize=24)
ax1.set_xlabel(bidi_text('فشار (bar)'), fontsize=24)
ax1.set_ylabel(bidi_text('دما (°C)'), fontsize=24)
ax1.grid(alpha=0.2)
ax1.legend(fontsize=21)

# افزودن معادله خط رگرسیون
slope = linear_model.coef_[0]
intercept = linear_model.intercept_
eq_text = bidi_text(f'معادله: دما = {slope:.2f} × فشار + {intercept:.2f}')
ax1.text(0.5, 0.05, eq_text, transform=ax1.transAxes,
        fontsize=24, color='red', ha='center',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# 2. نمودار رگرسیون غیرخطی
ax2.scatter(time, wear, color='green', alpha=0.7, label=bidi_text('داده‌های واقعی'))
ax2.plot(time, poly_pred, 'r-', linewidth=3, label=bidi_text('رگرسیون غیرخطی (درجه ۲)'))
ax2.set_title(bidi_text('رگرسیون غیرخطی: رابطه زمان کارکرد و سایش تجهیز'), fontsize=24)
ax2.set_xlabel(bidi_text('زمان کارکرد (ماه)'), fontsize=24)
ax2.set_ylabel(bidi_text('میزان سایش (mm)'), fontsize=24)
ax2.grid(alpha=0.2)
ax2.legend(fontsize=21)

# افزودن معادله رگرسیون غیرخطی
coefs = poly_model.named_steps['linearregression'].coef_
intercept = poly_model.named_steps['linearregression'].intercept_
eq_text = bidi_text(f'معادله: سایش = {coefs[2]:.3f} × زمان² + {coefs[1]:.3f} × زمان + {intercept:.2f}')
ax2.text(0.5, 0.05, eq_text, transform=ax2.transAxes,
        fontsize=24, color='red', ha='center',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# ایجاد بارکد
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=5, border=2)
qr.add_data("https://B2n.ir/ng3874")
qr.make(fit=True)

# تبدیل بارکد به تصویر
img = qr.make_image(fill_color="black", back_color="white")
img = img.convert("RGBA")
img_np = np.array(img)

# ایجاد جعبه برای بارکد با تنظیمات صحیح
imagebox = OffsetImage(img_np, zoom=0.8)  
ab = AnnotationBbox(
    imagebox, 
    (0.45, 0.2),  
    xycoords='figure fraction',  # استفاده از مختصات شکل اصلی
    box_alignment=(1, 0), 
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به محور فعلی
ax2.add_artist(ab)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(top=0.90, wspace=0.15)
plt.savefig('fig2_16.png', dpi=300, bbox_inches='tight')
plt.show()
