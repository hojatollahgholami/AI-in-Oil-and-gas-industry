import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import qrcode
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# تنظیمات فارسی‌نویسی
plt.rcParams["font.family"] = 'Adobe Arabic'
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.size'] = 20

# تابع برای نمایش متن فارسی
def persian_text(text):
    reshaped = reshape(text)
    return get_display(reshaped)

# مرحله ۱: داده‌های فرضی واقعی (x, y, cost)
np.random.seed(0)
x_data = np.random.uniform(-5, 5, 50)
y_data = np.random.uniform(-5, 5, 50)
# فرض تابع هزینه واقعی: ترکیب چندجمله‌ای درجه 2 + نویز
cost_data = 3 + 2*x_data + y_data**2 - 0.5*x_data*y_data + np.random.normal(0, 2, 50)

# مرحله ۲: برازش مدل رگرسیون چندجمله‌ای درجه 2
X_train = np.vstack([x_data, y_data]).T
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
model = LinearRegression().fit(X_poly, cost_data)

# مرحله ۳: ساخت مش برای ترسیم سطح
x_range = np.linspace(-6, 6, 100)
y_range = np.linspace(-6, 6, 100)
X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
XY_mesh = np.c_[X_mesh.ravel(), Y_mesh.ravel()]
Z_mesh = model.predict(poly.transform(XY_mesh)).reshape(X_mesh.shape)

# مرحله ۴: تعریف تابع هزینه و گرادیان‌ها برای بهینه‌سازی
def cost_function(x, y):
    input_poly = poly.transform([[x, y]])
    return model.predict(input_poly)[0]

def gradient(x, y):
    eps = 1e-4
    dx = (cost_function(x + eps, y) - cost_function(x - eps, y)) / (2 * eps)
    dy = (cost_function(x, y + eps) - cost_function(x, y - eps)) / (2 * eps)
    return np.array([dx, dy])

def hessian(x, y):
    eps = 1e-4
    dxx = (cost_function(x + eps, y) - 2*cost_function(x, y) + cost_function(x - eps, y)) / (eps**2)
    dyy = (cost_function(x, y + eps) - 2*cost_function(x, y) + cost_function(x, y - eps)) / (eps**2)
    dxy = (cost_function(x + eps, y + eps) - cost_function(x + eps, y - eps) -
           cost_function(x - eps, y + eps) + cost_function(x - eps, y - eps)) / (4 * eps**2)
    return np.array([[dxx, dxy], [dxy, dyy]])

# مرحله ۵: گرادیان کاهشی
path_gd = []
x, y = -4.5, -4.5
for _ in range(30):
    path_gd.append((x, y, cost_function(x, y)))
    grad = gradient(x, y)
    x, y = x - 0.1 * grad[0], y - 0.1 * grad[1]

# مرحله ۶: روش نیوتن
path_nt = []
x, y = -4.5, -4.5
for _ in range(10):
    path_nt.append((x, y, cost_function(x, y)))
    grad = gradient(x, y)
    hess = hessian(x, y)
    try:
        delta = np.linalg.solve(hess, grad)
        x, y = x - delta[0], y - delta[1]
    except np.linalg.LinAlgError:
        break  # اگر هسین منفرد بود، ادامه نده

# تبدیل مسیرها به آرایه برای ترسیم
path_gd = np.array(path_gd)
path_nt = np.array(path_nt)

# مرحله ۷: ترسیم نهایی
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# سطح هزینه
ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='viridis', alpha=0.7)

# نقاط داده اصلی
ax.scatter(x_data, y_data, cost_data, c='r', marker='x', label=persian_text('داده ها'))

# مسیر گرادیان کاهشی
ax.plot(path_gd[:,0], path_gd[:,1], path_gd[:,2], color='blue', marker='^', label=persian_text('گرادیان کاهشی'))

# مسیر نیوتن
ax.plot(path_nt[:,0], path_nt[:,1], path_nt[:,2], color='orange', marker='+', label=persian_text('روش نیوتن'))

ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.set_zlabel(persian_text("هزینه حفاری (میلیون دلار)"))
ax.legend()

# ایجاد بارکد
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=5, border=2)
qr.add_data("https://B2n.ir/tz5729")
qr.make(fit=True)

# تبدیل بارکد به تصویر
img = qr.make_image(fill_color="black", back_color="white")
img = img.convert("RGBA")
img_np = np.array(img)

# ایجاد جعبه برای بارکد با تنظیمات صحیح
imagebox = OffsetImage(img_np, zoom=0.8)  
ab = AnnotationBbox(
    imagebox, 
    (0.65, 0.17),  
    xycoords='figure fraction',  # استفاده از مختصات شکل اصلی
    box_alignment=(1, 0), 
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به محور فعلی
ax.add_artist(ab)

plt.tight_layout()
plt.savefig('fig2_20.png', dpi=300, bbox_inches='tight')
plt.show()
