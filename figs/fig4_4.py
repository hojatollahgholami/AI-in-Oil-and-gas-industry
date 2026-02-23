import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import qrcode

plt.rcParams['font.family'] = 'Adobe Arabic'
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.size'] = 18

# تابع کمکی برای نمایش صحیح متون فارسی
def fa(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# ایجاد شکل
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
fig.suptitle(fa('پیوند هوش مصنوعی و مدیریت انبار: کاهش هزینه‌ها با پیش‌بینی تقاضا'),
             fontsize=24, fontweight='bold', color='#1a3d7c')

# --- نمودار جریان فرآیند (سمت چپ) ---
ax1.set_title(fa('فرآیند سفارش، نگهداری و تحویل قطعات'), fontsize=18, pad=20)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

# رنگ‌ها
colors = {
    'start': '#4e79a7',
    'process': '#59a14f',
    'decision': '#f28e2c',
    'end': '#e15759',
    'data': '#76b7b2',
    'ai': '#af7aa1'
}

# عناصر فرآیند
processes = [
    (5, 9, fa('درخواست قطعه برای تعمیر'), 'process'),
    (5, 7, fa('بررسی موجودی انبار'), 'decision'),
    (2, 5, fa('سفارش به تامین کننده'), 'process'),
    (8, 5, fa('تحویل قطعه به متقاضی'), 'process'),
    (5, 3, fa('به‌روزرسانی موجودی انبار'), 'data'),
    (2, 3, fa('تحویل به انبار'), 'process'),
    (5, 1, fa('ثبت در سیستم نگهداری'), 'end'),
    (8, 7, fa('موجود'), 'decision'),
    (2, 7, fa('ناموجود'), 'decision')
]

# رسم عناصر
for x, y, text, ptype in processes:
    color = colors[ptype]

    if ptype == 'process':
        ax1.add_patch(patches.Rectangle((x-1.5, y-0.5), 3, 1, color=color, alpha=0.9))
        ax1.text(x, y, text, ha='center', va='center', fontsize=18)
    elif ptype == 'decision':
        ax1.add_patch(patches.Polygon([(x-1.5, y), (x, y+0.8), (x+1.5, y), (x, y-0.8)],
                       color=color, alpha=0.9))
        ax1.text(x, y, text, ha='center', va='center', fontsize=18)
    elif ptype == 'data':
        ax1.add_patch(patches.Rectangle((x-1.5, y-0.5), 3, 1, color=color, alpha=0.9))
        ax1.text(x, y, text, ha='center', va='center', fontsize=18)
    elif ptype == 'end':
        ax1.add_patch(patches.Circle((x, y), 0.7, color=color, alpha=0.9))
        ax1.text(x, y, text, ha='center', va='center', fontsize=18)

# فلش‌ها و اتصالات
connections = [
    (5, 9, 5, 8, ''),
    (5, 7.8, 5, 7.2, ''),
    (5, 7, 8, 7, fa('بله')),
    (5, 7, 2, 7, fa('خیر')),
    (2, 7, 2, 5.2, ''),
    (2, 5, 2, 3.8, ''),
    (2, 3, 5, 3.2, ''),
    (5, 3, 5, 1.8, ''),
    (8, 7, 8, 5.2, ''),
    (8, 5, 5, 5, fa('گزارش مصرف')),
    (5, 5, 5, 3.2, '')
]

for x1, y1, x2, y2, label in connections:
    ax1.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color='#555555'))
    if label:
        ax1.text((x1+x2)/2, (y1+y2)/2, label, ha='center', va='center',
                backgroundcolor='white', fontsize=16)

# افزودن نقش هوش مصنوعی
ai_box = patches.Rectangle((0.5, 0.5), 9, 3, fill=False, linestyle='--',
                          edgecolor=colors['ai'], linewidth=2)
ax1.add_patch(ai_box)
ax1.text(8, 3.2, fa('پیش‌بینی هوش مصنوعی'), color=colors['ai'],
        fontsize=18, fontweight='bold')
ax1.text(8, 2.5, fa('• تحلیل مصرف تاریخی\n• پیش‌بینی تقاضای آینده\n• تعیین سطح بهینه موجودی'),
        fontsize=16, va='top')

# --- نمودار پیش‌بینی XGBoost (سمت راست) ---
ax2.set_title(fa('پیش‌بینی تقاضای قطعات با الگوریتم XGBoost'), fontsize=18, pad=20)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# نمودار فرآیند پیش‌بینی
stages = [
    (1.5, 8, fa('داده‌های تاریخی'), '#4e79a7'),
    (4, 8, fa('پیش‌پردازش داده‌ها'), '#59a14f'),
    (6.5, 8, fa('مهندسی ویژگی‌ها'), '#f28e2c'),
    (9, 8, fa('مدل XGBoost'), '#e15759'),
    (9, 6, fa('پیش‌بینی تقاضا'), '#af7aa1'),
    (6.5, 4, fa('بهینه‌سازی موجودی'), '#76b7b2'),
    (4, 4, fa('کاهش هزینه‌ها'), '#59a14f'),
    (1.5, 4, fa('گزارش مدیریتی'), '#4e79a7')
]

# رسم مراحل
for x, y, text, color in stages:
    ax2.add_patch(patches.Rectangle((x-1.2, y-0.4), 2.4, 0.8, color=color, alpha=0.9))
    ax2.text(x, y, text, ha='center', va='center', fontsize=18, color='white')

# اتصالات
connections = [
    (1.5, 7.6, 4, 7.6),
    (4, 7.6, 6.5, 7.6),
    (6.5, 7.6, 9, 7.6),
    (9, 7.6, 9, 6.4),
    (9, 5.6, 6.5, 4.4),
    (6.5, 3.6, 4, 4.4),
    (4, 3.6, 1.5, 4.4)
]

for i, (x1, y1, x2, y2) in enumerate(connections):
    ax2.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="->", lw=2, color='#555555'))
    if i == 4:  # فلش بازخورد
        ax2.text((x1+x2)/2, (y1+y2)/2, fa('بازخورد'),
                ha='center', va='center', backgroundcolor='white', fontsize=18)

# نمودار نمونه پیش‌بینی
ax2.text(5, 2, fa('نمونه خروجی پیش‌بینی XGBoost'),
        ha='center', va='center', fontsize=18, fontweight='bold')

# داده‌های نمونه
months = [fa('فروردین'), fa('اردیبهشت'), fa('خرداد'), fa('تیر'), fa('مرداد'), fa('شهریور')]
actual = [120, 135, 115, 145, 130, 125]
predicted = [None, None, None, 140, 128, 132]

# رسم نمودار
ax2.plot(months, actual, 'o-', label=fa('مصرف واقعی'), linewidth=2, markersize=8, color='#4e79a7')
ax2.plot(months[3:], predicted[3:], 's--', label=fa('پیش‌بینی XGBoost'),
        linewidth=2, markersize=8, color='#e15759')

# خط نقطه‌چی برای پیش‌بینی
ax2.plot([months[2], months[3]], [actual[2], predicted[3]], 'k--', alpha=0.5)

# منطقه پیش‌بینی
ax2.fill_between(months[3:], [a-10 for a in actual[3:]], [a+10 for a in actual[3:]],
                color='#af7aa1', alpha=0.2)

ax2.set_ylabel(fa('تعداد قطعات'), fontsize=18)
ax2.legend(loc='lower center', fontsize=18)
ax2.grid(True, linestyle='--', alpha=0.3)

# باکس نتایج
results = [
    fa('کاهش ۳۰٪ موجودی انبار'),
    fa('کاهش ۴۵٪ کمبود قطعات'),
    fa('کاهش ۳۵٪ هزینه‌های نگهداری')
]

ax2.text(8, 1, fa('نتایج پیاده‌سازی:'), fontsize=18, fontweight='bold', ha='right')
for i, res in enumerate(results):
    ax2.text(8, 0.7-i*0.3, f"• {res}", fontsize=18, ha='right')

# افزودن توضیح پایانی
plt.figtext(0.5, 0.02,
           fa('پیش‌بینی تقاضا با XGBoost با دقت ۹۲٪ منجر به بهینه‌سازی موجودی و کاهش ۳۵٪ هزینه‌های انبارداری شد'),
           ha='center', fontsize=18, fontstyle='italic', color='#1a3d7c',
           bbox=dict(boxstyle='round', facecolor='#e0e0ff', alpha=0.7))

# -----------------------
# QR Code
# -----------------------

qr = qrcode.QRCode(box_size=3, border=2)
qr.add_data("https://B2n.ir/fig4_4")
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white").convert("RGBA")

img_np = np.array(img)
imagebox = OffsetImage(img_np, zoom=1.2)
ab = AnnotationBbox(imagebox,
                    (0.07, 0.8),
                    xycoords='figure fraction',
                    frameon=False)

fig.add_artist(ab)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('fig4_4.png', dpi=300, bbox_inches='tight')
plt.show()