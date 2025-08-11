
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.visualization.scatter import Scatter
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

# =============================================================================
# تعریف تابع هدف بر اساس داده‌های واقعی (cost و -production)
# =============================================================================
class DrillingProblem(Problem):
    def __init__(self, cost_fun, prod_fun):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=-5, xu=5)
        self.cost_fun = cost_fun
        self.prod_fun = prod_fun

    def _evaluate(self, X, out, *args, **kwargs):
        cost = self.cost_fun(X)
        production = self.prod_fun(X)
        out["F"] = np.column_stack([cost, -production])  # minimize both

# =============================================================================
# توابع هزینه و برداشت برحسب x و y
# =============================================================================
def cost_function(X):
    x, y = X[:,0], X[:,1]
    noise = np.random.normal(0, 0.2, len(x))
    return 10 + 0.5*x**2 + 0.3*y**2 + noise

def prod_function(X):
    x, y = X[:,0], X[:,1]
    return 5 + 2*np.exp(-0.1*(x-1)**2 - 0.2*(y+2)**2) + np.random.normal(0, 0.1, len(x))

problem = DrillingProblem(cost_function, prod_function)

# =============================================================================
# ثبت مسیر تکامل جمعیت
# =============================================================================
class MyCallback(Callback):
    def __init__(self):
        super().__init__()
        self.data["pop"] = []

    def notify(self, algorithm):
        self.data["pop"].append(algorithm.pop.get("F"))

callback = MyCallback()

# =============================================================================
# اجرای NSGA-II
# =============================================================================
algorithm = NSGA2(pop_size=100)
res = minimize(problem,
               algorithm,
               termination=get_termination("n_gen", 40),
               callback=callback,
               seed=1,
               verbose=False)

# =============================================================================
# ترسیم مسیر تکامل جمعیت
# =============================================================================
colors = ['gray', 'blue', 'green', 'red']
gens = [0, 10, 20, -1]  # نسل‌های انتخابی: اولیه، میانه، نهایی

fig, ax = plt.subplots(figsize=(8,6))
for i, gen in enumerate(gens):
    pop = callback.data["pop"][gen]
    plt.scatter(pop[:,0], pop[:,1],
                label=persian_text(f"نسل {gen if gen != -1 else len(callback.data['pop'])-1}"),
                alpha=0.6, s=50, color=colors[i], edgecolor='k')

ax.set_xlabel(persian_text("هزینه حفاری"))
ax.set_ylabel(persian_text("منفی برداشت"))
ax.legend()
ax.grid(True)

# ایجاد بارکد
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=5, border=2)
qr.add_data("https://B2n.ir/ew2772")
qr.make(fit=True)

# تبدیل بارکد به تصویر
img = qr.make_image(fill_color="black", back_color="white")
img = img.convert("RGBA")
img_np = np.array(img)

# ایجاد جعبه برای بارکد با تنظیمات صحیح
imagebox = OffsetImage(img_np, zoom=0.8)  
ab = AnnotationBbox(
    imagebox, 
    (0.7, 0.13),  
    xycoords='figure fraction',  # استفاده از مختصات شکل اصلی
    box_alignment=(1, 0), 
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به محور فعلی
ax.add_artist(ab)

fig.tight_layout()
fig.savefig('fig2_21.png', dpi=300, bbox_inches='tight')
plt.show()
