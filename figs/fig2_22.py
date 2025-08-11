!pip install SALib 

import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display
import qrcode
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# تنظیمات فارسی‌نویسی
plt.rcParams["font.family"] = "Adobe Arabic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.size'] = 18

def persian_text(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

# =============================================================================
# مدل سود تولید بنزین
# =============================================================================
def refinery_profit(params):
    """
    محاسبه سود تولید بنزین بر اساس پارامترهای ورودی
    پارامترها:
    - قیمت نفت خام (دلار/بشکه)
    - هزینه پالایش (دلار/لیتر)
    - قیمت فروش بنزین (دلار/لیتر)
    - حجم تولید (لیتر/روز)
    - هزینه ثابت (دلار/روز)
    - بازده تبدیل (%)
    - نرخ مالیات (%)
    """
    crude_price, refining_cost, selling_price, production_volume, fixed_cost, conversion_yield, tax_rate = params

    # محاسبه درآمد
    revenue = selling_price * production_volume * (conversion_yield / 100)

    # محاسبه هزینه‌ها
    crude_cost = (crude_price / 159) * production_volume  # تبدیل بشکه به لیتر
    variable_cost = refining_cost * production_volume
    total_cost = crude_cost + variable_cost + fixed_cost

    # محاسبه سود قبل از مالیات
    profit_before_tax = revenue - total_cost

    # محاسبه سود پس از مالیات
    profit = profit_before_tax * (1 - tax_rate / 100)

    return profit

# =============================================================================
# تحلیل حساسیت سراسری (روش Sobol)
# =============================================================================
# تعریف پارامترهای ورودی و محدوده تغییرات
problem = {
    'num_vars': 7,
    'names': [
        'crude_price',
        'refining_cost',
        'selling_price',
        'production_volume',
        'fixed_cost',
        'conversion_yield',
        'tax_rate'
    ],
    'bounds': [
        [40, 100],    # قیمت نفت خام (دلار/بشکه)
        [0.1, 0.5],   # هزینه پالایش (دلار/لیتر)
        [0.8, 1.5],   # قیمت فروش بنزین (دلار/لیتر)
        [500000, 2000000],  # حجم تولید (لیتر/روز)
        [50000, 200000],    # هزینه ثابت (دلار/روز)
        [70, 95],      # بازده تبدیل (%)
        [10, 30]       # نرخ مالیات (%)
    ]
}

# تولید نمونه‌ها با روش Saltelli
N = 1000
param_values = saltelli.sample(problem, N, calc_second_order=True)

# ارزیابی مدل برای تمام نمونه‌ها
Y = np.array([refinery_profit(vals) for vals in param_values])

# محاسبه شاخص‌های Sobol
Si = sobol.analyze(problem, Y, print_to_console=False)

# =============================================================================
# ترسیم نتایج
# =============================================================================
# تبدیل نام پارامترها به فارسی
persian_names = [
    persian_text('قیمت نفت خام'),
    persian_text('هزینه پالایش'),
    persian_text('قیمت فروش بنزین'),
    persian_text('حجم تولید'),
    persian_text('هزینه ثابت'),
    persian_text('بازده تبدیل'),
    persian_text('نرخ مالیات')
]

# شاخص‌های حساسیت مرتبه اول
S1 = Si['S1']
# شاخص‌های حساسیت کل
ST = Si['ST']

# ترسیم نمودار
fig, ax = plt.subplots(figsize=(14, 8))

# موقعیت میله‌ها
x = np.arange(len(persian_names))

# رسم میله‌ها
width = 0.35
rects1 = ax.bar(x - width/2, S1, width, label=persian_text('حساسیت مرتبه اول (S1)'))
rects2 = ax.bar(x + width/2, ST, width, label=persian_text('حساسیت کل (ST)'))

# تنظیمات نمودار
ax.set_ylabel(persian_text('شاخص حساسیت'), fontsize=22)
ax.set_xticks(x)
ax.set_xticklabels(persian_names, fontsize=21)
ax.legend(fontsize=24)
ax.grid(axis='y', alpha=0.3)

# افزودن مقادیر عددی روی میله‌ها
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=18)

autolabel(rects1)
autolabel(rects2)

# ایجاد بارکد
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=5, border=2)
qr.add_data("https://B2n.ir/ky4825")
qr.make(fit=True)

# تبدیل بارکد به تصویر
img = qr.make_image(fill_color="black", back_color="white")
img = img.convert("RGBA")
img_np = np.array(img)

# ایجاد جعبه برای بارکد با تنظیمات صحیح
imagebox = OffsetImage(img_np, zoom=0.8)  
ab = AnnotationBbox(
    imagebox, 
    (0.2, 0.73),  
    xycoords='figure fraction',  # استفاده از مختصات شکل اصلی
    box_alignment=(1, 0), 
    frameon=False,
    pad=0
)

# اضافه کردن بارکد به محور فعلی
ax.add_artist(ab)

plt.tight_layout()
plt.savefig('fig2_22.png', dpi=300)
plt.show()
