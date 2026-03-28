"""
Stage I Implementation – Cell-DNA Dynamics Instability Detector
================================================================
Sections:
  1. Global Parameters
  2. Signal Construction
  3. Feature Extraction
  4. Normalization & Reference State
  5. Instability Functional ΔΦ
  6. Visualization
  7. Null Tests
  8. CSV Output & Regime Statistics
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import csv

# ─────────────────────────────────────────────────────────────────────────────
# 1. GLOBAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
fs = 50            # sampling frequency (Hz)
T = 180            # total duration (s)
N = 9000           # total samples  (fs * T)
W = 400            # window size (samples)
step = 1           # sliding step (samples)
wE = wI = wC = 1   # feature weights
epsilon = 1e-6     # small constant to avoid division by zero

t = np.arange(N) / fs  # time vector [0, T)

# Regime boundary indices
i_trans = int(60 * fs)    # 3000 – start of transition
i_unstab = int(120 * fs)  # 6000 – start of unstable

rng = np.random.default_rng(seed=42)

# ─────────────────────────────────────────────────────────────────────────────
# 2. SIGNAL CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def _slow_random_walk(n_samples, scale=0.05, seed=None):
    """Gaussian random walk, cumsum-based, normalized to [-1, 1]."""
    local_rng = np.random.default_rng(seed)
    steps = local_rng.normal(0, scale, n_samples)
    walk = np.cumsum(steps)
    walk -= walk.mean()
    max_abs = np.abs(walk).max()
    if max_abs > 0:
        walk /= max_abs
    return walk


def build_signal():
    signal = np.zeros(N)

    # --- Stable [0, 60 s) ---
    n_s = i_trans
    t_s = t[:n_s]
    noise_s = rng.normal(0, 0.05, n_s)
    signal[:n_s] = np.sin(2 * np.pi * 0.25 * t_s) + noise_s

    # --- Transition [60, 120 s) ---
    n_tr = i_unstab - i_trans
    t_tr = t[i_trans:i_unstab]
    eta_A = _slow_random_walk(n_tr, scale=0.02, seed=1)
    eta_f = _slow_random_walk(n_tr, scale=0.02, seed=2)
    noise_tr = rng.normal(0, 0.2, n_tr)
    signal[i_trans:i_unstab] = (
        (1 + 0.5 * eta_A) * np.sin(2 * np.pi * (0.25 + 0.1 * eta_f) * t_tr)
        + noise_tr
    )

    # --- Unstable [120, 180 s] ---
    n_un = N - i_unstab
    t_un = t[i_unstab:]
    f_un = rng.uniform(0.1, 0.6, n_un)
    noise_un = rng.normal(0, 1, n_un)
    signal[i_unstab:] = np.sin(2 * np.pi * f_un * t_un) + 0.8 * noise_un

    return signal


signal = build_signal()

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE EXTRACTION  (sliding window, step=1)
# ─────────────────────────────────────────────────────────────────────────────
# Number of windows – centre each window at sample i (i = W//2 .. N-W//2)
n_windows = N - W + 1   # windows indexed 0 .. n_windows-1
# Time stamp at the centre of each window
t_feat = t[W // 2: W // 2 + n_windows]


def spectral_entropy(window):
    """Shannon entropy of the normalized power spectrum."""
    fft_mag = np.abs(np.fft.rfft(window))
    power = fft_mag ** 2
    total = power.sum()
    if total == 0:
        return 0.0
    pk = power / total
    # Avoid log(0)
    pk = pk[pk > 0]
    return -np.sum(pk * np.log(pk))


def coherence_metric(window, eps=epsilon):
    """Phase coherence via Hilbert transform: 1 / (Var(Δφ) + ε)."""
    analytic = hilbert(window)
    phase = np.unwrap(np.angle(analytic))
    delta_phi = np.diff(phase)
    return 1.0 / (np.var(delta_phi) + eps)


E_raw = np.empty(n_windows)
I_raw = np.empty(n_windows)
C_raw = np.empty(n_windows)

for k in range(n_windows):
    win = signal[k: k + W]
    E_raw[k] = np.mean(win ** 2)
    I_raw[k] = spectral_entropy(win)
    C_raw[k] = coherence_metric(win)

# ─────────────────────────────────────────────────────────────────────────────
# 4. NORMALIZATION & REFERENCE STATE
# ─────────────────────────────────────────────────────────────────────────────
# Baseline windows = those whose centre falls in [0, 60 s)
baseline_mask = t_feat < 60.0

mu_E, sigma_E = E_raw[baseline_mask].mean(), E_raw[baseline_mask].std()
mu_I, sigma_I = I_raw[baseline_mask].mean(), I_raw[baseline_mask].std()
mu_C, sigma_C = C_raw[baseline_mask].mean(), C_raw[baseline_mask].std()

# Guard against zero std (unlikely but safe)
sigma_E = sigma_E if sigma_E > epsilon else epsilon
sigma_I = sigma_I if sigma_I > epsilon else epsilon
sigma_C = sigma_C if sigma_C > epsilon else epsilon

En = (E_raw - mu_E) / sigma_E
In = (I_raw - mu_I) / sigma_I
Cn = (C_raw - mu_C) / sigma_C

# Reference state x* = normalised baseline means = (0, 0, 0) by construction
E_star = 0.0
I_star = 0.0
C_star = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 5. INSTABILITY FUNCTIONAL ΔΦ(t)
# ─────────────────────────────────────────────────────────────────────────────
delta_phi = (
    wE * (En - E_star) ** 2
    + wI * (In - I_star) ** 2
    + wC * (Cn - C_star) ** 2
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALIZATION  (3-panel figure)
# ─────────────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
fig_path = os.path.join(script_dir, "stage_i_figure.png")

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# Panel 1 – raw signal
axes[0].plot(t, signal, lw=0.4, color="steelblue")
axes[0].set_ylabel("s(t)")
axes[0].set_title("Raw Signal")

# Panel 2 – features E(t) / I(t) / C(t) (normalised)
axes[1].plot(t_feat, En, lw=0.6, label="E(t)")
axes[1].plot(t_feat, In, lw=0.6, label="I(t)")
axes[1].plot(t_feat, Cn, lw=0.6, label="C(t)")
axes[1].set_ylabel("Normalised value")
axes[1].set_title("Features E(t), I(t), C(t)")
axes[1].legend(loc="upper left", fontsize=8)

# Panel 3 – ΔΦ(t)
axes[2].plot(t_feat, delta_phi, lw=0.6, color="crimson")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("ΔΦ(t)")
axes[2].set_title("Instability Functional ΔΦ(t)")

# Vertical regime lines on all panels
for ax in axes:
    ax.axvline(60, color="orange", lw=1.5, ls="--", label="t = 60 s")
    ax.axvline(120, color="red", lw=1.5, ls="--", label="t = 120 s")

# Show regime-line legend only on the last panel to avoid clutter
axes[2].legend(loc="upper left", fontsize=8)

plt.tight_layout()
fig.savefig(fig_path, dpi=300)
plt.close(fig)
print(f"Figure saved → {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. NULL TESTS
# ─────────────────────────────────────────────────────────────────────────────

def compute_delta_phi(sig):
    """Re-run feature extraction + ΔΦ pipeline on an arbitrary signal."""
    nw = len(sig) - W + 1
    E_ = np.empty(nw)
    I_ = np.empty(nw)
    C_ = np.empty(nw)
    for k in range(nw):
        w = sig[k: k + W]
        E_[k] = np.mean(w ** 2)
        I_[k] = spectral_entropy(w)
        C_[k] = coherence_metric(w)

    bm = t[W // 2: W // 2 + nw] < 60.0
    sE = E_[bm].std(); sE = sE if sE > epsilon else epsilon
    sI = I_[bm].std(); sI = sI if sI > epsilon else epsilon
    sC = C_[bm].std(); sC = sC if sC > epsilon else epsilon

    En_ = (E_ - E_[bm].mean()) / sE
    In_ = (I_ - I_[bm].mean()) / sI
    Cn_ = (C_ - C_[bm].mean()) / sC

    return En_ ** 2 + In_ ** 2 + Cn_ ** 2


def phase_randomize(sig, seed=0):
    """Return signal with randomised FFT phases (preserves power spectrum)."""
    local_rng = np.random.default_rng(seed)
    Sf = np.fft.rfft(sig)
    phases = local_rng.uniform(0, 2 * np.pi, len(Sf))
    Sf_rand = np.abs(Sf) * np.exp(1j * phases)
    return np.fft.irfft(Sf_rand, n=len(sig))


def time_shuffle(sig, seed=1):
    """Return randomly permuted version of signal."""
    local_rng = np.random.default_rng(seed)
    return local_rng.permutation(sig)


null_signals = {
    "Phase-randomised": phase_randomize(signal),
    "Time-shuffled": time_shuffle(signal),
}

print("\n── Null Test Results ──")
for label, null_sig in null_signals.items():
    dp_null = compute_delta_phi(null_sig)
    t_null = t[W // 2: W // 2 + len(dp_null)]
    bm_s = t_null < 60.0
    bm_tr = (t_null >= 60.0) & (t_null < 120.0)
    bm_un = t_null >= 120.0
    m_s = dp_null[bm_s].mean()
    m_tr = dp_null[bm_tr].mean()
    m_un = dp_null[bm_un].mean()
    ordered = m_s < m_tr < m_un
    print(
        f"  {label:22s}: "
        f"stable={m_s:.4f}  trans={m_tr:.4f}  unstable={m_un:.4f}  "
        f"ordered={ordered}  (expected False)"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 8. CSV OUTPUT & REGIME STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
csv_path = "stage_i_results.csv"
with open(csv_path, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["time", "E", "I", "C", "delta_phi"])
    for k in range(n_windows):
        writer.writerow([
            f"{t_feat[k]:.4f}",
            f"{E_raw[k]:.6f}",
            f"{I_raw[k]:.6f}",
            f"{C_raw[k]:.6f}",
            f"{delta_phi[k]:.6f}",
        ])
print(f"CSV saved → {csv_path}  ({n_windows} rows)")

# ── Regime statistics ──────────────────────────────────────────────────────
bm_stable = t_feat < 60.0
bm_trans  = (t_feat >= 60.0) & (t_feat < 120.0)
bm_unstab = t_feat >= 120.0

mean_stable = delta_phi[bm_stable].mean()
mean_trans  = delta_phi[bm_trans].mean()
mean_unstab = delta_phi[bm_unstab].mean()

print("\n── Regime ΔΦ Means ──")
print(f"  Stable    : {mean_stable:.4f}")
print(f"  Transition: {mean_trans:.4f}")
print(f"  Unstable  : {mean_unstab:.4f}")

ordering_ok = mean_stable < mean_trans < mean_unstab
print(f"\n  Ordering check (stable < transition < unstable): {ordering_ok}")
if not ordering_ok:
    print("  WARNING: Ordering NOT satisfied.")
