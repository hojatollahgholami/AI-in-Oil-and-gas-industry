pip install pymoo

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as minimize_moo
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display
import qrcode
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# تنظیمات فارسی‌نویسی
plt.rcParams["font.family"] = "B Nazanin"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.size'] = 18

def persian_text(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

# =============================================================================
# مسئله بهینه‌سازی انتخاب تجهیزات خط لوله گاز
# =============================================================================
class GasPipelineOptimization(Problem):
    def __init__(self):
        """
        بهینه سازی انتخاب تجهیزات برای خط لوله گاز
        متغیرهای تصمیم:
          x: قطر لوله (اینچ)
          y: فشار کاری (بار)
          z: ضخامت دیواره لوله (میلی متر)

        اهداف:
          1. حداقل سازی هزینه ساخت
          2. حداقل سازی تلفات انرژی
          3. حداکثرسازی ایمنی
        """
        super().__init__(n_var=3, n_obj=3, n_constr=0,
                         xl=np.array([10, 20, 5]),
                         xu=np.array([50, 100, 30]))

    def _evaluate(self, X, out, *args, **kwargs):
        diameter, pressure, thickness = X[:, 0], X[:, 1], X[:, 2]

        # تابع هدف 1: هزینه ساخت
        material_cost = 0.05 * diameter * thickness
        installation_cost = 0.02 * diameter**2
        f1 = material_cost + installation_cost

        # تابع هدف 2: تلفات انرژی
        # تلفات انرژی با فشار رابطه معکوس و با قطر رابطه مستقیم دارد
        energy_loss = 1000 / (pressure * np.sqrt(diameter))
        f2 = energy_loss

        # تابع هدف 3: ایمنی (منفی می کنیم چون می خواهیم ماکزیمم شود)
        # ایمنی با ضخامت رابطه مستقیم و با فشار رابطه معکوس دارد
        safety = (thickness / 5) * (100 / pressure)
        f3 = -safety  # منفی برای ماکزیمم سازی

        out["F"] = np.column_stack([f1, f2, f3])

# ایجاد و حل مسئله
problem = GasPipelineOptimization()
algorithm = NSGA2(pop_size=100)
res = minimize_moo(problem, algorithm, ('n_gen', 50), seed=1, verbose=False)

# استخراج نتایج
solutions = res.X
objectives = res.F

# جداسازی مقادیر
diameters = solutions[:, 0]
pressures = solutions[:, 1]
thicknesses = solutions[:, 2]
costs = objectives[:, 0]
energy_losses = objectives[:, 1]
safety = -objectives[:, 2]  # تبدیل به مثبت برای ایمنی واقعی

# =============================================================================
# ترسیم نمودارهای شفاف
# =============================================================================
fig = plt.figure(figsize=(18, 12))

# ساختار GridSpec برای چیدمان نمودارها
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1])

# نمودار 1: رابطه بین قطر لوله و فشار کاری
ax1 = fig.add_subplot(gs[0])
sc1 = ax1.scatter(diameters, pressures, c=costs, cmap='viridis', s=80, alpha=0.8)

# خطوط راهنما برای محدوده‌های ایمنی
ax1.axvline(20, color='r', linestyle='--', alpha=0.4)
ax1.axvline(40, color='r', linestyle='--', alpha=0.4)
ax1.axhline(40, color='g', linestyle='--', alpha=0.4)
ax1.axhline(80, color='g', linestyle='--', alpha=0.4)

ax1.set_xlabel(persian_text('قطر لوله (اینچ)'), fontsize=21)
ax1.set_ylabel(persian_text('فشار کاری (بار)'), fontsize=21)
ax1.set_title(persian_text('رابطه قطر لوله و فشار کاری با رنگ نشان دهنده هزینه'), fontsize=21)
ax1.grid(alpha=0.3)

cbar1 = plt.colorbar(sc1, ax=ax1)
cbar1.set_label(persian_text('هزینه ساخت (هزار دلار)'), fontsize=21)

# توضیحات مناطق
ax1.annotate(persian_text('منطقه ایمن'),
            xy=(30, 60), xytext=(15, 70),
            arrowprops=dict(arrowstyle='->'))
ax1.annotate(persian_text('هزینه بالا'),
            xy=(45, 90), xytext=(35, 70),
            arrowprops=dict(arrowstyle='->'))

# نمودار 2: رابطه بین ایمنی و تلفات انرژی
ax2 = fig.add_subplot(gs[1])
sc2 = ax2.scatter(safety, energy_losses, c=thicknesses, cmap='plasma', s=80)

# خط روند
z = np.polyfit(safety, energy_losses, 1)
p = np.poly1d(z)
ax2.plot(safety, p(safety), 'r--', alpha=0.7)

ax2.set_xlabel(persian_text('سطح ایمنی'), fontsize=21)
ax2.set_ylabel(persian_text('تلفات انرژی (کیلووات ساعت)'), fontsize=21)
ax2.set_title(persian_text('مبادله ایمنی و تلفات انرژی'), fontsize=21)
ax2.grid(alpha=0.3)

cbar2 = plt.colorbar(sc2, ax=ax2)
cbar2.set_label(persian_text('ضخامت دیواره (میلی متر)'), fontsize=21)

# نمودار 3: تاثیر ضخامت دیواره بر هزینه و ایمنی
ax3 = fig.add_subplot(gs[2])
sc3 = ax3.scatter(thicknesses, costs, c=safety, cmap='coolwarm', s=80)

# تقسیم بندی مناطق
ax3.fill_between([5, 15], 0, 15, color='green', alpha=0.1)
ax3.fill_between([15, 30], 0, 15, color='yellow', alpha=0.1)
ax3.fill_between([25, 30], 0, 15, color='red', alpha=0.1)

ax3.set_xlabel(persian_text('ضخامت دیواره (میلی‌متر)'), fontsize=21)
ax3.set_ylabel(persian_text('هزینه ساخت (هزار دلار)'), fontsize=21)
ax3.set_title(persian_text('تاثیر ضخامت دیواره بر هزینه و ایمنی'), fontsize=21)
ax3.grid(alpha=0.3)

cbar3 = plt.colorbar(sc3, ax=ax3)
cbar3.set_label(persian_text('سطح ایمنی'), fontsize=21)

# توضیحات مناطق
ax3.text(8, 3, persian_text('منطقه بهینه'), fontsize=21, color='green')
ax3.text(16, 3, persian_text('منطقه قابل قبول'), fontsize=21, color='orange')
ax3.text(25, 3, persian_text('هزینه بالا'), fontsize=21, color='red')

# نمودار 4: جبهه پارتو سه‌بعدی
ax4 = fig.add_subplot(gs[3], projection='3d')

# نقاط جبهه پارتو
sc4 = ax4.scatter(costs, energy_losses, safety,
                 c=thicknesses, cmap='viridis', s=50, alpha=0.8)

# برچسب‌های محورها
ax4.set_xlabel(persian_text('هزینه ساخت'), fontsize=21, labelpad=15)
ax4.set_ylabel(persian_text('تلفات انرژی'), fontsize=21, labelpad=15)
ax4.set_zlabel(persian_text('سطح ایمنی'), fontsize=21, labelpad=15)
ax4.set_title(persian_text('جبهه پارتو سه بعدی اهداف بهینه سازی'), fontsize=21)

# جهت‌های بهینه‌سازی
ax4.quiver(0, 0, 0, 15, 0, 0, color='r', arrow_length_ratio=0.1, label=persian_text('کاهش هزینه'))
ax4.quiver(0, 0, 0, 0, 100, 0, color='g', arrow_length_ratio=0.1, label=persian_text('کاهش تلفات'))
ax4.quiver(0, 0, 0, 0, 0, 100, color='b', arrow_length_ratio=0.1, label=persian_text('افزایش ایمنی'))

ax4.legend()
ax4.view_init(elev=20, azim=45)

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
    (0.16, 0.3),  
    xycoords='figure fraction',  # استفاده از مختصات شکل اصلی
    box_alignment=(1, 0), 
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به محور فعلی
ax3.add_artist(ab)

# تنظیم فاصله‌ها
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(wspace=0.25, hspace=0.3)

plt.savefig('fig2_20.png', dpi=300, bbox_inches='tight')
plt.show()
