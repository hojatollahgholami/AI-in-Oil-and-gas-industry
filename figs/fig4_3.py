import matplotlib.pyplot as plt
import numpy as np
import random
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import qrcode

plt.rcParams['font.family'] = 'Adobe Arabic'
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.size'] = 18

def fa(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# -----------------------
# تعریف داده‌ها
# -----------------------
teams = 2
tasks = [
    ("تعمیر کمپرسور", 4),
    ("تعمیر فن", 3),
    ("تعمیر پمپ", 5),
    ("تعویض فیلتر", 2),
    ("بازرسی شیر", 3),
    ("تنظیم ولو", 4),
    ("تعمیر موتور", 6)
]

durations = [t[1] for t in tasks]
task_names = [fa(t[0]) for t in tasks]
n = len(tasks)

# -----------------------
# الگوریتم ژنتیک
# -----------------------

POP_SIZE = 40
GENS = 100
MUT_RATE = 0.1

def fitness(chromosome):
    team_times = [0, 0]
    for i in range(n):
        team_times[chromosome[i]] += durations[i]
    return -max(team_times)   # کمینه کردن makespan

def random_chromosome():
    return [random.randint(0,1) for _ in range(n)]

def crossover(p1, p2):
    point = random.randint(1, n-1)
    return p1[:point] + p2[point:]

def mutate(ch):
    if random.random() < MUT_RATE:
        idx = random.randint(0, n-1)
        ch[idx] = 1 - ch[idx]
    return ch

# تولید جمعیت اولیه
population = [random_chromosome() for _ in range(POP_SIZE)]

for _ in range(GENS):
    population = sorted(population, key=fitness, reverse=True)
    new_pop = population[:10]  # elitism
    
    while len(new_pop) < POP_SIZE:
        p1, p2 = random.sample(population[:20], 2)
        child = crossover(p1, p2)
        child = mutate(child)
        new_pop.append(child)
        
    population = new_pop

best = sorted(population, key=fitness, reverse=True)[0]

# -----------------------
# استخراج برنامه نهایی
# -----------------------

team_times = [0,0]
after_schedule = []

for i in range(n):
    team = best[i]
    start = team_times[team]
    after_schedule.append((task_names[i], start, durations[i], team))
    team_times[team] += durations[i]

optimized_time = max(team_times)

# -----------------------
# برنامه اولیه غیربهینه (دو تیم ولی نامتوازن)
# -----------------------

# تخصیص دستی بد (مثال)
initial_assignment = [0,0,0,1,0,1,0]

team_times_before = [0,0]
before_schedule = []

for i in range(n):
    team = initial_assignment[i]
    start = team_times_before[team]
    before_schedule.append((task_names[i], start, durations[i], team))
    team_times_before[team] += durations[i]

initial_time = max(team_times_before)

reduction_percent = ((initial_time - optimized_time) / initial_time) * 100

# -----------------------
# رسم نمودار
# -----------------------
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='salmon', label=fa('تیم 1 (T1)')),
    Patch(facecolor='orange', label=fa('تیم 2 (T2)'))
]
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

axes[0].set_title(fa("برنامه اولیه"), pad=20)
for i, (name, start, duration, team) in enumerate(before_schedule):
    color = 'salmon' if team==0 else 'orange'
    axes[0].barh(i, duration, left=start, color=color)
    axes[0].text(start+duration/2, i,
                 f"{name}\n(T{team+1})",
                 ha='center', va='center')
    

    
    axes[0].legend(handles=legend_elements,
                   loc='right',
                   frameon=True)

axes[0].set_yticks([])
axes[0].grid(axis='x', linestyle='--', alpha=0.5)

axes[1].set_title(fa("برنامه بهینه‌شده با الگوریتم ژنتیک"), pad=20)

for i, (name, start, duration, team) in enumerate(after_schedule):
    color = 'skyblue' if team==0 else 'lightgreen'
    axes[1].barh(i, duration, left=start, color=color)
    axes[1].text(start+duration/2, i,
                 f"{name}\n(T{team+1})",
                 ha='center', va='center')

axes[1].set_yticks([])
axes[1].set_xlabel(fa("زمان (روز)"))
axes[1].grid(axis='x', linestyle='--', alpha=0.5)

fig.text(0.5, 0.15,
         fa(f'کاهش زمان کل: {initial_time:.1f} به {optimized_time:.1f} روز ({reduction_percent:.1f}%)'),
         ha='center', fontsize=18)

# -----------------------
# QR Code
# -----------------------

qr = qrcode.QRCode(box_size=3, border=2)
qr.add_data("https://B2n.ir/fig4_3")
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white").convert("RGBA")

img_np = np.array(img)
imagebox = OffsetImage(img_np, zoom=0.8)
ab = AnnotationBbox(imagebox,
                    (0.07, 0.28),
                    xycoords='figure fraction',
                    frameon=False)

fig.add_artist(ab)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('fig4_3.png', dpi=300)
plt.show()