"""
Extract critical exponents β and δ from the dataset.

β: Fit ln(ρ_ss) vs ln(Ω - Ω_c) for Ω > Ω_c. Slope = β.
   Reference: thesis β ≈ 0.586 for 2D N=3600.

δ: Fit ln(ρ(t)) vs ln(t) at Ω = Ω_c (algebraic decay ρ ~ t^{-δ}).
   Reference: thesis δ ≈ 0.4577 for 2D.

Also performs finite-size extrapolation of δ by fitting each N separately.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
from scipy import stats
from data.dataset_v2 import TrajectoryRecord

OMEGA_C = 11.2


def load_data():
    with open("outputs/rydberg_dataset_v2.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def extract_beta(records, omega_c):
    """
    Fit ln(ρ_ss) = β * ln(Ω - Ω_c) + const for Ω > Ω_c.
    Returns β and fit statistics.
    """
    omegas = []
    rho_ss_vals = []

    for r in records:
        if r.omega <= omega_c + 1e-6:
            continue
        # ρ_ss = abs(last point of rho)
        rho = np.abs((1 + r.sz_mean) / 2)
        rho_ss = np.abs(rho[-1])
        if rho_ss < 1e-12:
            continue
        omegas.append(r.omega)
        rho_ss_vals.append(rho_ss)

    if len(omegas) < 3:
        return None, None, None

    omegas = np.array(omegas)
    rho_ss_vals = np.array(rho_ss_vals)
    delta_omega = omegas - omega_c

    # Fit in log-log space
    log_domega = np.log(delta_omega)
    log_rho = np.log(rho_ss_vals)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_domega, log_rho)

    return slope, r_value ** 2, std_err


def extract_delta_at_critical(records_at_critical):
    """
    Fit ln(ρ(t)) = -δ * ln(t) + const at Ω = Ω_c.
    We fit the middle portion of the curve to avoid early transients and late saturation.
    """
    if not records_at_critical:
        return None, None, None

    r = records_at_critical[0]
    valid = r.t_save <= 2000
    t = r.t_save[valid]
    sz = r.sz_mean[valid]
    rho = np.abs((1 + sz) / 2)

    # Avoid t=0 and very late times
    # Fit in the range where decay is algebraic (typically t ∈ [10, 200] or similar)
    mask = (t > 5) & (t < 500) & (rho > 1e-12)
    if mask.sum() < 5:
        mask = (t > 1) & (rho > 1e-12)

    t_fit = t[mask]
    rho_fit = rho[mask]

    log_t = np.log(t_fit)
    log_rho = np.log(rho_fit)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_t, log_rho)
    delta = -slope

    return delta, r_value ** 2, std_err


def extract_delta_by_fit_range(records_at_critical, t_min=5, t_max=500):
    """Extract δ with a specified fit range."""
    if not records_at_critical:
        return None, None, None

    r = records_at_critical[0]
    valid = r.t_save <= 2000
    t = r.t_save[valid]
    sz = r.sz_mean[valid]
    rho = np.abs((1 + sz) / 2)

    mask = (t >= t_min) & (t <= t_max) & (rho > 1e-12)
    if mask.sum() < 5:
        return None, None, None

    log_t = np.log(t[mask])
    log_rho = np.log(rho[mask])

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_t, log_rho)
    delta = -slope

    return delta, r_value ** 2, std_err


def main():
    data = load_data()

    print("=" * 60)
    print("Critical Exponent Extraction — 2D, γ=0.1")
    print("=" * 60)

    # Group by N
    from collections import defaultdict
    by_n = defaultdict(list)
    for r in data:
        if r.dimension != 2 or r.gamma != 0.1:
            continue
        by_n[r.n_atoms].append(r)

    # ── β extraction per N ──────────────────────────────────────────
    print("\n--- β extraction (ln ρ_ss vs ln(Ω - Ω_c)) ---")
    print(f"{'N':>8}  {'β':>10}  {'R²':>10}  {'stderr':>10}")
    print("-" * 45)
    beta_by_n = {}
    for n in sorted(by_n.keys()):
        beta, r2, stderr = extract_beta(by_n[n], OMEGA_C)
        if beta is not None:
            beta_by_n[n] = beta
            print(f"{n:8d}  {beta:10.4f}  {r2:10.4f}  {stderr:10.4f}")
        else:
            print(f"{n:8d}  {'insufficient data':>30}")

    if beta_by_n:
        print(f"\nβ values: mean={np.mean(list(beta_by_n.values())):.4f}, "
              f"std={np.std(list(beta_by_n.values())):.4f}")

    # ── δ extraction per N (at Ω_c) ─────────────────────────────────
    print("\n--- δ extraction (algebraic decay at Ω_c) ---")
    print(f"{'N':>8}  {'δ':>10}  {'R²':>10}  {'stderr':>10}  {'fit range':>15}")
    print("-" * 55)

    delta_by_n = {}
    for n in sorted(by_n.keys()):
        # Find record at Ω_c
        at_critical = [r for r in by_n[n] if abs(r.omega - OMEGA_C) < 1e-3]

        # Try different fit ranges
        best_delta, best_r2, best_range = None, -1, ""
        for t_min, t_max in [(5, 100), (10, 200), (10, 500), (20, 300), (5, 500)]:
            delta, r2, stderr = extract_delta_by_fit_range(at_critical, t_min, t_max)
            if delta is not None and r2 > best_r2:
                best_delta, best_r2, best_range = delta, r2, f"[{t_min}, {t_max}]"

        if best_delta is not None:
            delta_by_n[n] = best_delta
            print(f"{n:8d}  {best_delta:10.4f}  {best_r2:10.4f}  {'---':>10}  {best_range:>15}")
        else:
            # Fallback: auto range
            delta, r2, stderr = extract_delta_at_critical(at_critical)
            if delta is not None:
                delta_by_n[n] = delta
                print(f"{n:8d}  {delta:10.4f}  {r2:10.4f}  {stderr:10.4f}  {'auto':>15}")
            else:
                print(f"{n:8d}  {'insufficient data':>40}")

    if delta_by_n:
        deltas = list(delta_by_n.values())
        print(f"\nδ values: mean={np.mean(deltas):.4f}, std={np.std(deltas):.4f}")

        # Finite-size extrapolation of δ (thesis: δ converges to ~0.45 for large N)
        if len(delta_by_n) >= 3:
            ns = np.array(sorted(delta_by_n.keys()), dtype=float)
            deltas_arr = np.array([delta_by_n[int(n)] for n in ns])
            # Fit δ(N) = δ_∞ + a/N
            inv_n = 1.0 / ns
            slope, intercept, r_value, p_value, std_err = stats.linregress(inv_n, deltas_arr)
            print(f"\nFinite-size extrapolation: δ(∞) = {intercept:.4f} (R²={r_value**2:.4f})")
            print(f"  δ(N) = {intercept:.4f} + {slope:.4f}/N")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary (thesis values for comparison)")
    print("=" * 60)
    print(f"β (thesis, 2D N=3600):  0.586 ± 0.002")
    print(f"δ (thesis, 2D):         0.4577 (converges to ~0.45)")
    print(f"z (thesis, 2D small N): 1.86")
    print(f"z (thesis, 2D large N): ~2.2")
    print("=" * 60)


if __name__ == "__main__":
    main()
