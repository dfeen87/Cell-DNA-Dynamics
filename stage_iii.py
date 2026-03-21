import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert, savgol_filter
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score, roc_curve

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "fs": 50,
    "T": 180.0,
    "t1": 60.0,   # stable -> pre
    "t2": 120.0,  # pre -> unstable
    "window_sec": 8.0,
    "smooth_window": 101,  # plotting only
    "smooth_poly": 3,
    "n_seeds": 50,
    "lead_threshold_quantile": 0.90,  # threshold from stable segment only
}

# ============================================================
# 1. SYNTHETIC TRIADIC SYSTEM
# ============================================================
def simulate_signal(seed: int, cfg: dict):
    rng = np.random.default_rng(seed)
    fs = cfg["fs"]
    T = cfg["T"]
    t1 = cfg["t1"]
    t2 = cfg["t2"]

    t = np.arange(0, T, 1.0 / fs)
    x = np.zeros_like(t)

    # Ground-truth labels
    # 0 = stable, 1 = pre-instability, 2 = instability
    y_regime = np.zeros_like(t, dtype=int)
    y_regime[(t >= t1) & (t < t2)] = 1
    y_regime[t >= t2] = 2

    # Build a signal where:
    # - E changes via amplitude drift
    # - I changes via spectral complexity / burstiness
    # - C changes via phase irregularity
    phase_acc = 0.0

    for i, ti in enumerate(t):
        if ti < t1:
            # Stable
            A = 1.0 + 0.02 * rng.normal()
            f = 0.25 + 0.003 * rng.normal()
            phase_jitter = 0.0 + 0.002 * rng.normal()
            noise_sigma = 0.04
            burst = 0.0

        elif ti < t2:
            # Pre-instability
            tau = ti - t1
            A = 1.0 + 0.004 * tau + 0.03 * rng.normal()
            f = 0.25 + 0.0015 * tau + 0.005 * rng.normal()
            phase_jitter = 0.01 + 0.01 * (tau / (t2 - t1)) + 0.01 * rng.normal()
            noise_sigma = 0.10 + 0.001 * tau
            # low-amplitude intermittent burst component
            burst = 0.15 * np.sin(2 * np.pi * 1.8 * ti) * (rng.random() < 0.05)

        else:
            # Instability
            tau = ti - t2
            A = 1.3 + 0.006 * tau + 0.05 * rng.normal()
            f = 0.35 + 0.03 * np.sin(0.2 * ti) + 0.01 * rng.normal()
            phase_jitter = 0.08 + 0.03 * rng.normal()
            noise_sigma = 0.30
            burst = 0.35 * rng.normal() * (rng.random() < 0.10)

        phase_acc += 2 * np.pi * f / fs + phase_jitter
        x[i] = A * np.sin(phase_acc) + rng.normal(0, noise_sigma) + burst

    return t, x, y_regime


# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================
def spectral_entropy(segment: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(segment))
    spec = spec + 1e-12
    p = spec / np.sum(spec)
    return float(entropy(p))


def extract_features(x: np.ndarray, cfg: dict):
    fs = cfg["fs"]
    window = int(cfg["window_sec"] * fs)
    n = len(x)

    E = np.full(n, np.nan)
    I = np.full(n, np.nan)
    C = np.full(n, np.nan)

    analytic = hilbert(x)
    phase = np.unwrap(np.angle(analytic))

    for i in range(window, n):
        seg = x[i - window:i]

        # Energy proxy
        E[i] = np.sqrt(np.mean(seg ** 2))

        # Information proxy
        I[i] = spectral_entropy(seg)

        # Coherence proxy: inverse phase-difference variance
        ph = phase[i - window:i]
        dph = np.diff(ph)
        C[i] = 1.0 / (np.var(dph) + 1e-6)

    return E, I, C, window


# ============================================================
# 3. NORMALIZATION + DELTAPHI
# ============================================================
def zscore_from_baseline(arr: np.ndarray, baseline_mask: np.ndarray):
    mu = np.nanmean(arr[baseline_mask])
    sd = np.nanstd(arr[baseline_mask]) + 1e-8
    return (arr - mu) / sd, mu, sd


def compute_delta_phi(E, I, C, y_regime, weights=(1.0, 1.0, 1.0)):
    # Use only stable regime as baseline
    baseline_mask = (y_regime == 0)

    E_n, E_mu, E_sd = zscore_from_baseline(E, baseline_mask)
    I_n, I_mu, I_sd = zscore_from_baseline(I, baseline_mask)
    C_n, C_mu, C_sd = zscore_from_baseline(C, baseline_mask)

    wE, wI, wC = weights
    delta = np.sqrt(wE * E_n**2 + wI * I_n**2 + wC * C_n**2)

    return {
        "E_n": E_n, "I_n": I_n, "C_n": C_n,
        "DeltaPhi": delta,
        "baseline_stats": {
            "E": (E_mu, E_sd), "I": (I_mu, I_sd), "C": (C_mu, C_sd)
        }
    }


# ============================================================
# 4. METRICS
# ============================================================
def stable_threshold(score: np.ndarray, y_regime: np.ndarray, q: float):
    stable_vals = score[y_regime == 0]
    stable_vals = stable_vals[np.isfinite(stable_vals)]
    return float(np.quantile(stable_vals, q))


def compute_lead_time(t: np.ndarray, score: np.ndarray, y_regime: np.ndarray, threshold: float, t2: float):
    valid = np.isfinite(score)
    idx = np.where(valid & (t < t2) & (score >= threshold))[0]
    if len(idx) == 0:
        return np.nan
    first_cross = t[idx[0]]
    return t2 - first_cross


def binary_auc(score: np.ndarray, y_true_binary: np.ndarray):
    valid = np.isfinite(score)
    if len(np.unique(y_true_binary[valid])) < 2:
        return np.nan
    return roc_auc_score(y_true_binary[valid], score[valid])


# ============================================================
# 5. BASELINES
# ============================================================
def compute_baselines(E_n, I_n, C_n):
    # Single-metric baselines
    return {
        "E_only": np.abs(E_n),
        "I_only": np.abs(I_n),
        "C_only": np.abs(C_n),
    }


# ============================================================
# 6. ABLATION
# ============================================================
def compute_ablation(E_n, I_n, C_n):
    return {
        "DeltaPhi_full": np.sqrt(E_n**2 + I_n**2 + C_n**2),
        "DeltaPhi_EI": np.sqrt(E_n**2 + I_n**2),
        "DeltaPhi_EC": np.sqrt(E_n**2 + C_n**2),
        "DeltaPhi_IC": np.sqrt(I_n**2 + C_n**2),
    }


# ============================================================
# 7. ONE-RUN VISUALIZATION
# ============================================================
def plot_one_run(t, x, y_regime, E_n, I_n, C_n, delta, cfg, title_suffix="", save_path=None):
    delta_s = savgol_filter(np.nan_to_num(delta, nan=0.0), cfg["smooth_window"], cfg["smooth_poly"])

    colors = {
        0: "#cfe8ff",  # stable
        1: "#f7e7b7",  # pre
        2: "#f8c9cf",  # instability
    }

    fig, axs = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    # raw
    axs[0].plot(t, x, lw=1.0)
    axs[0].set_title(f"Raw Signal {title_suffix}")
    axs[0].set_ylabel("x(t)")

    # features
    axs[1].plot(t, E_n, label="E_n(t)", lw=1.0)
    axs[1].plot(t, I_n, label="I_n(t)", lw=1.0)
    axs[1].plot(t, C_n, label="C_n(t)", lw=1.0)
    axs[1].legend(loc="upper right")
    axs[1].set_title("Normalized Features")
    axs[1].set_ylabel("z-score")

    # raw DeltaPhi
    axs[2].plot(t, delta, color="crimson", lw=1.0)
    axs[2].set_title("DeltaPhi(t)")
    axs[2].set_ylabel("DeltaPhi")

    # smoothed for visualization
    axs[3].plot(t, delta_s, color="purple", lw=1.5)
    axs[3].set_title("Smoothed DeltaPhi(t) (visualization only)")
    axs[3].set_ylabel("DeltaPhi_s")
    axs[3].set_xlabel("Time (s)")

    # regime shading
    for ax in axs:
        for regime_val in [0, 1, 2]:
            mask = (y_regime == regime_val)
            if np.any(mask):
                # contiguous spans
                idx = np.where(mask)[0]
                starts = [idx[0]]
                ends = []
                for i in range(1, len(idx)):
                    if idx[i] != idx[i - 1] + 1:
                        ends.append(idx[i - 1])
                        starts.append(idx[i])
                ends.append(idx[-1])

                for s, e in zip(starts, ends):
                    ax.axvspan(t[s], t[e], color=colors[regime_val], alpha=0.4)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================
# 8. MULTI-SEED BENCHMARK
# ============================================================
def run_benchmark(cfg: dict):
    results = []

    for seed in range(cfg["n_seeds"]):
        t, x, y_regime = simulate_signal(seed, cfg)
        E, I, C, window = extract_features(x, cfg)

        valid_mask = np.arange(len(t)) >= window

        out = compute_delta_phi(E, I, C, y_regime)
        E_n = out["E_n"]
        I_n = out["I_n"]
        C_n = out["C_n"]
        delta = out["DeltaPhi"]

        baselines = compute_baselines(E_n, I_n, C_n)
        ablation = compute_ablation(E_n, I_n, C_n)

        # binary task: instability vs non-instability
        y_bin = (y_regime == 2).astype(int)

        # threshold from stable regime only
        thr = stable_threshold(delta[valid_mask], y_regime[valid_mask], cfg["lead_threshold_quantile"])
        lead = compute_lead_time(t[valid_mask], delta[valid_mask], y_regime[valid_mask], thr, cfg["t2"])

        row = {
            "seed": seed,
            "threshold": thr,
            "lead_time": lead,
            "auc_full": binary_auc(delta, y_bin),
            "auc_E_only": binary_auc(baselines["E_only"], y_bin),
            "auc_I_only": binary_auc(baselines["I_only"], y_bin),
            "auc_C_only": binary_auc(baselines["C_only"], y_bin),
            "auc_EI": binary_auc(ablation["DeltaPhi_EI"], y_bin),
            "auc_EC": binary_auc(ablation["DeltaPhi_EC"], y_bin),
            "auc_IC": binary_auc(ablation["DeltaPhi_IC"], y_bin),
            "mean_stable": np.nanmean(delta[(y_regime == 0) & valid_mask]),
            "mean_pre": np.nanmean(delta[(y_regime == 1) & valid_mask]),
            "mean_instability": np.nanmean(delta[(y_regime == 2) & valid_mask]),
        }
        results.append(row)

        # plot first seed only
        if seed == 0:
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage_iii_seed0.png")
            plot_one_run(t[valid_mask], x[valid_mask], y_regime[valid_mask],
                         E_n[valid_mask], I_n[valid_mask], C_n[valid_mask], delta[valid_mask], cfg,
                         title_suffix="(seed 0)", save_path=save_path)

    return results


# ============================================================
# 9. SUMMARY
# ============================================================
def summarize_results(results):
    keys = [
        "lead_time", "auc_full", "auc_E_only", "auc_I_only", "auc_C_only",
        "auc_EI", "auc_EC", "auc_IC",
        "mean_stable", "mean_pre", "mean_instability"
    ]

    # ---- per-seed CSV rows ----
    csv_rows = []
    for r in results:
        csv_rows.append({
            "seed": r["seed"],
            "threshold": r["threshold"],
            "lead_time": r["lead_time"],
            "auc_full": r["auc_full"],
            "auc_E_only": r["auc_E_only"],
            "auc_I_only": r["auc_I_only"],
            "auc_C_only": r["auc_C_only"],
            "auc_EI": r["auc_EI"],
            "auc_EC": r["auc_EC"],
            "auc_IC": r["auc_IC"],
            "mean_stable": r["mean_stable"],
            "mean_pre": r["mean_pre"],
            "mean_instability": r["mean_instability"],
        })

    # ---- check ordering per seed ----
    stable = np.array([r["mean_stable"] for r in results], dtype=float)
    pre = np.array([r["mean_pre"] for r in results], dtype=float)
    instab = np.array([r["mean_instability"] for r in results], dtype=float)
    ordering_ok = np.mean((stable < pre) & (pre < instab))

    # ---- summary row (mean ± std for each metric + ordering rate) ----
    summary_row = {"seed": "mean ± std", "threshold": f"ordering_rate={100*ordering_ok:.1f}%"}
    for k in keys:
        vals = np.array([r[k] for r in results], dtype=float)
        summary_row[k] = f"{np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}"
    csv_rows.append(summary_row)

    # ---- save CSV ----
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage_iii_summary.csv")
    df = pd.DataFrame(csv_rows)
    col_order = ["seed", "threshold", "lead_time", "auc_full",
                 "auc_E_only", "auc_I_only", "auc_C_only",
                 "auc_EI", "auc_EC", "auc_IC",
                 "mean_stable", "mean_pre", "mean_instability"]
    df = df[col_order]
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # ---- print summary table ----
    print("\n=== Multi-seed summary ===")
    for k in keys:
        vals = np.array([r[k] for r in results], dtype=float)
        print(f"{k:22s}: mean={np.nanmean(vals):.4f}, std={np.nanstd(vals):.4f}")

    # ---- print ordering rate ----
    print(f"\nOrdering stable < pre < instability holds in {100*ordering_ok:.1f}% of seeds.")

    # ---- print AUC comparison ----
    auc_keys = ["auc_full", "auc_E_only", "auc_I_only", "auc_C_only", "auc_EI", "auc_EC", "auc_IC"]
    print("\n=== AUC comparison (mean ± std) ===")
    for k in auc_keys:
        vals = np.array([r[k] for r in results], dtype=float)
        marker = " <-- full ΔΦ" if k == "auc_full" else ""
        print(f"{k:22s}: {np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}{marker}")

    full_auc = np.nanmean([r["auc_full"] for r in results])
    e_auc = np.nanmean([r["auc_E_only"] for r in results])
    i_auc = np.nanmean([r["auc_I_only"] for r in results])
    c_auc = np.nanmean([r["auc_C_only"] for r in results])
    outperforms = full_auc > e_auc and full_auc > i_auc and full_auc > c_auc
    print(f"\nAUC_full > AUC_E_only, AUC_I_only, AUC_C_only: {outperforms}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    results = run_benchmark(CONFIG)
    summarize_results(results)
