import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# تابع لود داده‌ها (با پیاده‌سازی ایمن)
def load_primary_series(csv_file, country, product, flow, unit):
    df = pd.read_csv(csv_file, dtype=str)
    sel = df.query(
        "REF_AREA == @country and ENERGY_PRODUCT == @product and FLOW_BREAKDOWN == @flow and UNIT_MEASURE == @unit"
    ).copy()
    if sel.empty:
        raise ValueError(f"No rows found for {country}/{product}/{flow}/{unit} in {csv_file}")


    # تبدیل مقادیر به عدد، تبدیل تاریخ
    sel["OBS_VALUE"] = pd.to_numeric(sel["OBS_VALUE"], errors="coerce")
    sel["TIME_PERIOD"] = pd.to_datetime(sel["TIME_PERIOD"], errors="coerce")


    sel = sel.dropna(subset=["TIME_PERIOD", "OBS_VALUE"])
    if sel.empty:
        raise ValueError("No numeric date/value rows after cleaning.")


    series = sel.sort_values("TIME_PERIOD").set_index("TIME_PERIOD")["OBS_VALUE"].astype(float)
    # حذف مقادیر غیرمثبت (در صورت نیاز) — بعضی داده‌ها صفر/منفی دارند که لگاریتم/نسبت را خراب می‌کنند
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if (series <= 0).any():
        # اگر تعدادی صفر/منفی داریم، آنها را حذف می‌کنیم (یا می‌توانیم مقدار کوچک جایگزین کنیم)
        series = series[series > 0]
    return series


# ---------------------------
# ساخت حالات و ماتریس گذار مارکوف با ایمنی
def compute_states_and_transition(series, desired_states=3, laplace_alpha=1.0):
    # استفاده از بازده درصدی (pct_change*100) تا مستقل از مقیاس باشد
    dy = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna() * 100.0
    if dy.size < 2:
        raise ValueError("Series too short after computing returns; need more observations.")


    # تعداد حالات واقعی (اگر داده تغییرپذیری کمی دارد، تعداد حالات را کاهش می‌دهیم)
    unique_vals = np.unique(np.round(dy, 8))
    n_states = min(desired_states, unique_vals.size) if unique_vals.size >= 2 else 0
    if n_states < 2:
        raise ValueError("Not enough variation to build at least 2 states. Try another series or aggregate.")


    # بنا بر quantile‌ها تقسیم می‌کنیم
    qs = np.linspace(0, 1, n_states + 1)
    bins = np.quantile(dy, qs)


    # اگر لبه‌ها برابرند (بدون تنوع)، کمی جابجا می‌کنیم تا pd.cut کار کند
    for i in range(1, len(bins)):
        if bins[i] <= bins[i-1]:
            bins[i] = bins[i-1] + 1e-8


    state_series = pd.cut(dy, bins=bins, labels=False, include_lowest=True)
    state_series = state_series.astype(int)


    # شمارش گذارها
    counts = np.zeros((n_states, n_states), dtype=float)
    for a, b in zip(state_series[:-1], state_series[1:]):
        counts[a, b] += 1.0


    # Laplace smoothing (برای جلوگیری از سطرهای صفر)
    counts = counts + laplace_alpha
    row_sums = counts.sum(axis=1, keepdims=True)
    P = counts / row_sums  # ماتریس گذار


    # پارامترهای هر حالت (میانگین و انحراف معیار بازده%)
    mu_by_state = {}
    sd_by_state = {}
    overall_mu = dy.mean()
    overall_sd = dy.std(ddof=1) if dy.std(ddof=1) > 0 else 1e-3


    grouped = dy.groupby(state_series)
    for st in range(n_states):
        if st in grouped.groups:
            vals = grouped.get_group(st).values
            mu_by_state[st] = np.nanmean(vals)
            sd_by_state[st] = np.nanstd(vals, ddof=1)
            if np.isnan(sd_by_state[st]) or sd_by_state[st] <= 0:
                sd_by_state[st] = overall_sd * 0.5 + 0.01
        else:
            mu_by_state[st] = overall_mu
            sd_by_state[st] = overall_sd


    return state_series, P, mu_by_state, sd_by_state, dy


# ---------------------------
# شبیه‌سازی مارکوف + مونت‌کارلو ایمن
def simulate_markov_montecarlo(series, state_series, P, mu_by_state, sd_by_state,
                               horizon=24, n_sim=1000, seed=42, sample_paths_to_plot=50):
    np.random.seed(seed)
    last_level = series.iloc[-1]
    last_state = int(state_series.iloc[-1])  # ممکن است IndexError اگر state_series خالی باشد


    n_states = P.shape[0]
    sims = np.full((n_sim, horizon), np.nan)
    for sim in range(n_sim):
        level = last_level
        state = last_state
        for t in range(horizon):
            probs = P[state]
            # ایمنی: اگر به خاطر عددی مشکل داشته باشیم، احتمال یکنواخت بگذار
            if np.any(np.isnan(probs)) or probs.sum() <= 0:
                probs = np.ones(n_states) / n_states
            next_state = np.random.choice(np.arange(n_states), p=probs)
            mu = mu_by_state.get(next_state, 0.0)
            sd = sd_by_state.get(next_state, 0.1)
            # نمونه‌گیری درصد تغییر (بازده%)
            d = np.random.normal(mu, sd)
            # به‌روزرسانی سطح به صورت ضربی (مناسب برای درصدها)
            level = level * np.exp(d / 100.0)
            sims[sim, t] = level
            state = next_state


    # تبدیل به آمار خلاصه
    p05 = np.percentile(sims, 5, axis=0)
    p10 = np.percentile(sims, 10, axis=0)
    p25 = np.percentile(sims, 25, axis=0)
    p50 = np.percentile(sims, 50, axis=0)
    p75 = np.percentile(sims, 75, axis=0)
    p90 = np.percentile(sims, 90, axis=0)
    p95 = np.percentile(sims, 95, axis=0)


    return sims, {"p05": p05, "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90, "p95": p95}


# ---------------------------
# رسم نتایج
def plot_results(series, sims, stats, horizon, title=None, sample_paths=50):
    last_date = series.index[-1]
    dates_future = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")


    plt.figure(figsize=(12,6))
    plt.plot(series.index, series.values, label="Historical", color="black", linewidth=2)


    # رسم تعدادی مسیر نمونه
    sample_paths = min(sample_paths, sims.shape[0])
    for i in range(sample_paths):
        plt.plot(dates_future, sims[i, :], color="lightblue", alpha=0.25)


    # میانه و باندها
    plt.plot(dates_future, stats["p50"], label="Median", color="blue", linewidth=2)
    plt.fill_between(dates_future, stats["p05"], stats["p95"], alpha=0.15, label="5%-95% band")
    plt.fill_between(dates_future, stats["p25"], stats["p75"], alpha=0.25, label="25%-75% band")


    plt.legend()
    plt.grid(True)
    if title:
        plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Production")
    plt.tight_layout()
    plt.show()


# ---------------------------
# Main: پارامترها و اجرا
if __name__ == "__main__":
    # مسیر و پارامترها را اینجا تنظیم کنید:
    csv_file = "NewProcedure_Primary_CSV.csv"   # فایل extracted از world_primary_csv.zip
    country = "IR"        # مثال: "AE" امارات
    product = "CRUDEOIL"  # یا "NGL" ، "OTHERCRUDE" ...
    flow = "INDPROD"      # یا "PRODUCTION" یا "INDPROD" بسته به فایل شما (بررسی کنید)
    unit = "KBD"          # واحد: "KBD" یا "KBBL" یا غیره
    horizon = 24
    n_sim = 1000
    seed = 42


    # لود داده
    series = load_primary_series(csv_file, country, product, flow, unit)
    print(f"Loaded series for {country}/{product}/{flow}/{unit} with {len(series)} observations.")
    print("Recent data:\n", series.tail(6))


    # محاسبه حالات و ماتریس گذار
    state_series, P, mu_by_state, sd_by_state, dy = compute_states_and_transition(series, desired_states=3, laplace_alpha=1.0)
    print("\nTransition matrix (P):")
    print(pd.DataFrame(P).round(3))
    print("\nState mu, sd (percent changes):")
    for st in sorted(mu_by_state):
        print(f" state {st}: mu={mu_by_state[st]:.4f}%, sd={sd_by_state[st]:.4f}%")


    # شبیه‌سازی
    sims, stats = simulate_markov_montecarlo(series, state_series, P, mu_by_state, sd_by_state,
                                             horizon=horizon, n_sim=n_sim, seed=seed)


    # رسم
    plot_results(series, sims, stats, horizon, title=f"{country} {product} Markov+MonteCarlo forecast ({horizon}m)")
