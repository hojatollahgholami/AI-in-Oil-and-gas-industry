import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import qrcode

plt.rcParams['font.family'] = 'Adobe Arabic'
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.size'] = 26

# ---------- helper ----------
def fa(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(22, 10))
ax.set_xlim(0, 24)
ax.set_ylim(0, 16)
ax.axis('off')

# ---------- steps (snake layout) ----------
steps = [

    # row 1 (left → right)
    (3,14, fa("شروع: وقوع خرابی"), 'ellipse', '#43A047'),
    (9,14, fa("تعریف و اعتبارسنجی مشکل"), 'rectangle', '#64B5F6'),
    (15,14, fa("AI: تشخیص ناهنجاری"), 'rectangle', '#81C784'),
    (21,14, fa("جمع‌آوری و یکپارچه‌سازی داده‌ها"), 'rectangle', '#64B5F6'),

    # row 2 (right → left)
    (21,11.5, fa("AI: تجمیع داده‌های چندمنبعی"), 'rectangle', '#81C784'),
    (15,11.5, fa("تحلیل علل بالقوه"), 'rectangle', '#FFB74D'),
    (9,11.5, fa("AI: کشف روابط علّی"), 'rectangle', '#81C784'),
    (3,11.5, fa("اولویت‌بندی علت ریشه‌ای"), 'rectangle', '#FFB74D'),

    # row 3 (left → right)
    (3,9, fa("AI: رتبه‌بندی علل"), 'rectangle', '#81C784'),
    (9,9, fa("تولید و ارزیابی راهکارها"), 'rectangle', '#BA68C8'),
    (15,9, fa("AI: پیشنهاد اقدام اصلاحی"), 'rectangle', '#81C784'),
    (21,9, fa("پیاده‌سازی و پایش اثربخشی"), 'rectangle', '#64B5F6'),

    # decision row
    (21,6.5, fa("آیا مشکل برطرف شد؟"), 'diamond', '#E57373'),

    # bottom
    (15,6.5, fa("استانداردسازی و ثبت دانش"), 'rectangle', '#64B5F6'),
    (9,6.5, fa("AI: به‌روزرسانی پایگاه دانش"), 'rectangle', '#81C784'),
    (3,6.5, fa("پایان"), 'ellipse', '#43A047'),
]

# ---------- draw shapes ----------
for x, y, text, shape, color in steps:
    if shape == 'rectangle':
        ax.add_patch(patches.Rectangle((x-2.2, y-0.7), 4.4, 1.4,
                      facecolor=color, edgecolor='black', alpha=0.92))
        plt.text(x, y, text, ha='center', va='center')

    elif shape == 'diamond':
        ax.add_patch(patches.Polygon(
            [(x, y+1.0), (x+2.2, y), (x, y-1.0), (x-2.2, y)],
            facecolor=color, edgecolor='black', alpha=0.92))
        plt.text(x, y, text, ha='center', va='center')

    elif shape == 'ellipse':
        ax.add_patch(patches.Ellipse((x, y), 4.2, 1.4,
                      facecolor=color, edgecolor='black', alpha=0.95))
        plt.text(x, y, text, ha='center', va='center', fontweight='bold')

# ---------- arrow helper ----------
def arrow(x1,y1,x2,y2,label=None):
    ax.annotate("",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.8))
    if label:
        plt.text((x1+x2)/2, (y1+y2)/2+0.25, label,
                 ha='center', fontsize=18,
                 bbox=dict(boxstyle='round,pad=0.2',
                           facecolor='white', alpha=0.9))

# ---------- snake connections ----------
# row1
arrow(5.2,14,6.8,14)
arrow(11.2,14,12.8,14)
arrow(17.2,14,18.8,14)

# down right
arrow(21,13.3,21,12.3)

# row2 (reverse)
arrow(18.8,11.5,17.2,11.5)
arrow(12.8,11.5,11.2,11.5)
arrow(6.8,11.5,5.2,11.5)

# down left
arrow(3,10.8,3,9.8)

# row3
arrow(5.2,9,6.8,9)
arrow(11.2,9,12.8,9)
arrow(17.2,9,18.8,9)

# to decision
arrow(21,8.3,21,7.5)

# decision branches
arrow(19,6.5,17.2,6.5,label=fa("بله"))
arrow(19,6.5,17.2,11.5,label=fa("خیر"))

# bottom flow
arrow(12.8,6.5,11.2,6.5)
arrow(6.8,6.5,5.2,6.5)

# ---------- QR ----------
qr = qrcode.QRCode(box_size=3, border=2)
qr.add_data("https://B2n.ir/fig4_5")
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white").convert("RGBA")

img_np = np.array(img)
imagebox = OffsetImage(img_np, zoom=1.4)
ab = AnnotationBbox(imagebox, (0.07, 0.45),
                    xycoords='figure fraction',
                    frameon=False)
#fig.add_artist(ab)

plt.tight_layout()
plt.savefig('fig4_5.png', dpi=300, bbox_inches='tight')
plt.show()