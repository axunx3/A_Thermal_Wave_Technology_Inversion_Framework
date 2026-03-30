import torch
import math
import numpy as np
import pandas as pd
import os
from scipy.optimize import differential_evolution
from functools import partial

# ════════════════════════════════════════════════════════════════
# Physical / instrument constants
# ════════════════════════════════════════════════════════════════
b = 5e-7            # heater half-width [m]
D_OMEGA = 125.528e-9  # ω = D_OMEGA / depth²  (instrument constant)
p0 = 0.94           # heat flux amplitude per unit length [W/m]

# ════════════════════════════════════════════════════════════════
# Hyperparameters  (all universal — no per-case tuning)
# ════════════════════════════════════════════════════════════════
S0 = 5              # baseline window size (number of initial sweep steps)
Z_TAU = 3.0         # detection multiplier (dimensionless)
EPSILON = 0.05      # relative floor for σ_Δ
D_FLOOR = 1e-8      # noise-floor gate for d_F*

DE_POPSIZE = 30     # DE population size
DE_MAXITER = 500    # DE max generations
DE_TOL = 1e-6       # DE convergence tolerance
DE_F = 0.8          # DE scaling factor
DE_CR = 0.9         # DE crossover probability


# ════════════════════════════════════════════════════════════════
# Numerical utilities
# ════════════════════════════════════════════════════════════════

def safe_exp(z):
    zr = torch.clamp(z.real, min=-320, max=320)
    return torch.exp(zr + 1j * z.imag)

def safe_sinh(z):
    return 0.5 * safe_exp(z) - 0.5 * safe_exp(-z)

def safe_cosh(z):
    return 0.5 * safe_exp(z) + 0.5 * safe_exp(-z)


# ════════════════════════════════════════════════════════════════
# Discrete Fréchet distance
# ════════════════════════════════════════════════════════════════

try:
    from numba import njit
    @njit(cache=True)
    def _frechet_dp(P, Q):
        """Numba-accelerated Fréchet DP core (~50x faster than pure Python)."""
        m, n = len(P), len(Q)
        CA = np.empty((m, n), np.float64)
        CA[0, 0] = abs(P[0] - Q[0])
        for i in range(1, m):
            CA[i, 0] = max(CA[i - 1, 0], abs(P[i] - Q[0]))
        for j in range(1, n):
            CA[0, j] = max(CA[0, j - 1], abs(P[0] - Q[j]))
        for i in range(1, m):
            for j in range(1, n):
                CA[i, j] = max(min(CA[i - 1, j], CA[i, j - 1], CA[i - 1, j - 1]),
                               abs(P[i] - Q[j]))
        return CA[m - 1, n - 1]
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _frechet_dp_python(P, Q):
    """Pure-Python fallback (iterative DP, no recursion)."""
    m, n = len(P), len(Q)
    CA = np.empty((m, n), dtype=np.float64)
    CA[0, 0] = abs(P[0] - Q[0])
    for i in range(1, m):
        CA[i, 0] = max(CA[i - 1, 0], abs(P[i] - Q[0]))
    for j in range(1, n):
        CA[0, j] = max(CA[0, j - 1], abs(P[0] - Q[j]))
    for i in range(1, m):
        for j in range(1, n):
            CA[i, j] = max(min(CA[i - 1, j], CA[i, j - 1], CA[i - 1, j - 1]),
                           abs(P[i] - Q[j]))
    return float(CA[m - 1, n - 1])


def discrete_frechet(P, Q):
    if isinstance(P, torch.Tensor):
        P = P.detach().cpu().numpy()
    if isinstance(Q, torch.Tensor):
        Q = Q.detach().cpu().numpy()
    P = np.asarray(P, dtype=np.float64).ravel()
    Q = np.asarray(Q, dtype=np.float64).ravel()
    m, n = len(P), len(Q)
    if m == 0 or n == 0:
        return 0.0
    if _HAS_NUMBA:
        return float(_frechet_dp(P, Q))
    return _frechet_dp_python(P, Q)


# ════════════════════════════════════════════════════════════════
# Forward models  (1-layer, 2-layer, 3-layer)
# ════════════════════════════════════════════════════════════════
# Pre-compute integration grids ONCE (the biggest repeated cost).
_J = torch.complex(torch.tensor(0.0), torch.tensor(1.0))

# 1-layer grid (high resolution for Stage 1 accuracy)
_M1      = torch.logspace(math.log10(1e-9), math.log10(1e7), 100000).reshape(-1, 1).double()
_M1_flat = _M1[:, 0]
_SIN1    = (torch.sin(_M1 * b) ** 2 / (_M1 * b) ** 2)

# 2/3-layer grid (coarser — called thousands of times inside DE)
_M2      = torch.logspace(math.log10(1e-9), math.log10(1e7), 10000).reshape(-1, 1).double()
_M2_flat = _M2[:, 0]
_SIN2    = (torch.sin(_M2 * b) ** 2 / (_M2 * b) ** 2)

# 2/3-layer grid (coarser — called thousands of times inside DE)
_M3      = torch.logspace(math.log10(1e-9), math.log10(1e7), 7500).reshape(-1, 1).double()
_M3_flat = _M3[:, 0]
_SIN3    = (torch.sin(_M3 * b) ** 2 / (_M3 * b) ** 2)


def compute_T_1L(k1, omega_vec, CV1):
    """Single-layer ΔT_{2ω}(ω)."""
    D1 = k1 / CV1
    omega = omega_vec.reshape(1, -1)
    u1 = torch.sqrt(_M1 ** 2 + _J * omega / D1)
    integrand = (p0 / (torch.pi * k1)) * _SIN1 / u1
    return torch.trapezoid(integrand, _M1_flat, dim=0).real


def compute_T_2L(k1, D1, omega_vec, L1, k2, CV2):
    """Two-layer ΔT_{2ω}(ω)."""
    D2 = k2 / CV2
    omega = omega_vec.reshape(1, -1)
    m2 = _M2  # alias for readability

    u1 = torch.sqrt(m2 ** 2 + _J * omega / D1)
    u2 = torch.sqrt(m2 ** 2 + _J * omega / D2)
    gamma1, gamma2 = u1 * k1, u2 * k2

    epos = safe_exp(u1 * L1)
    eneg = safe_exp(-u1 * L1)
    B_plus  = (gamma1 + gamma2) * epos + (gamma1 - gamma2) * eneg
    B_minus = (gamma1 + gamma2) * epos - (gamma1 - gamma2) * eneg
    integrand = (p0 / (torch.pi * k1)) * _SIN2 / u1 * (B_plus / B_minus)

    return torch.trapezoid(integrand, _M2_flat, dim=0).real


def compute_T_3L(k1, D1, k2, D2, L1, omega_vec, L2, k3, CV3):
    """Three-layer ΔT_{2ω}(ω)."""
    D3 = k3 / CV3
    omega = omega_vec.reshape(1, -1)
    m2 = _M3

    u1 = torch.sqrt(m2 ** 2 + _J * omega / D1)
    u2 = torch.sqrt(m2 ** 2 + _J * omega / D2)
    u3 = torch.sqrt(m2 ** 2 + _J * omega / D3)
    gamma1, gamma2, gamma3 = u1 * k1, u2 * k2, u3 * k3

    nu1, nu2 = u1 * L1, u2 * L2
    sh1, ch1 = safe_sinh(nu1), safe_cosh(nu1)
    sh2, ch2 = safe_sinh(nu2), safe_cosh(nu2)

    A = gamma3 * sh2 + gamma2 * ch2
    B = gamma3 * ch2 + gamma2 * sh2
    num = gamma1 * ch1 * A + gamma2 * sh1 * B
    den = gamma1 * sh1 * A + gamma2 * ch1 * B
    integrand = (p0 / (torch.pi * k1)) * _SIN3 / u1 * (num / (den + 1e-12))

    return torch.trapezoid(integrand, _M3_flat, dim=0).real


# ════════════════════════════════════════════════════════════════
# ATCS computation
# ════════════════════════════════════════════════════════════════

def compute_atcs(T_data, omega_data):
    ln_omega = torch.log(omega_data)
    dT_dlnw = torch.gradient(T_data, spacing=(ln_omega,), dim=0, edge_order=1)[0]
    return -p0 / (2 * math.pi * dT_dlnw)


def atcs_proxy(T):
    return 1.0 / (T[1:] - T[:-1])


# ════════════════════════════════════════════════════════════════
# Loss functions for DE  (L2 norm on ATCS proxy)
# ════════════════════════════════════════════════════════════════

def _log_atcs_l2(kapp_exp, kapp_mod):
    exp = torch.log(kapp_exp[2:-2])
    mod = torch.log(kapp_mod[2:-2])
    return float(torch.mean((exp - mod) ** 2))


def loss_2L(params, T_exp, omega_vec, k1, CV1):
    """Stage 2 objective: fit 2-layer model to T_exp (L2 on log-ATCS)."""
    L1, k2, CV2 = params
    D1 = k1 / CV1
    T_mod = compute_T_2L(k1, D1, omega_vec, L1, k2, CV2)
    kapp_exp = compute_atcs(T_exp, omega_vec)
    kapp_mod = compute_atcs(T_mod, omega_vec)
    return _log_atcs_l2(kapp_exp, kapp_mod)


def loss_3L(params, T_exp, omega_vec, k1, CV1, L1, k2, CV2):
    """Stage 3 objective: fit 3-layer model to T_exp (L2 on log-ATCS)."""
    L2, k3, CV3 = params
    D1, D2 = k1 / CV1, k2 / CV2
    T_mod = compute_T_3L(k1, D1, k2, D2, L1, omega_vec, L2, k3, CV3)
    kapp_exp = compute_atcs(T_exp, omega_vec)
    kapp_mod = compute_atcs(T_mod, omega_vec)
    return _log_atcs_l2(kapp_exp, kapp_mod)


# ════════════════════════════════════════════════════════════════
# Self-calibrating plateau-departure criterion
# ════════════════════════════════════════════════════════════════

def detect_mutation(current_val, history, s0=S0, z_tau=Z_TAU,
                    epsilon=EPSILON, d_floor=D_FLOOR, bilateral=False):
    if len(history) < s0:
        return False

    # Noise-floor gate (only for Fréchet loss, not ATCS)
    if not bilateral and current_val < d_floor:
        return False

    # Rolling baseline: use the MOST RECENT s0 elements
    recent = history[-s0:]
    deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
    mu = sum(deltas) / len(deltas)
    var = sum((d - mu) ** 2 for d in deltas) / (len(deltas) - 1)
    sigma = max(var ** 0.5, epsilon * abs(mu))
    if sigma == 0.0:
        sigma = 1e-15

    # Current first difference
    delta_s = current_val - history[-1]

    # Standardized departure
    z = (delta_s - mu) / sigma
    return abs(z) > z_tau if bilateral else z > z_tau


# ════════════════════════════════════════════════════════════════
# Stage 1 — Superficial layer extraction via ATCS plateau
# ════════════════════════════════════════════════════════════════

def stage1(T_data, omega_data, s0=S0, z_tau=Z_TAU):
    k_app = compute_atcs(T_data, omega_data)
    k_np = k_app.numpy()

    # Skip index 0 (gradient edge artifact)
    scan_start = 1
    history = []
    n_mut = None

    for n in range(scan_start, len(k_np)):
        val = float(k_np[n])
        if detect_mutation(val, history, s0=s0, z_tau=z_tau, bilateral=True):
            n_mut = n
            break
        history.append(val)

    if n_mut is not None:
        # Boundary truncation: go back s0 steps from mutation point
        boundary_idx = max(scan_start + s0, n_mut - s0)
    else:
        # No departure → entire spectrum is single-layer
        boundary_idx = len(k_np) - 1

    # k1 = mean of the stable plateau
    k1 = float(k_np[scan_start:boundary_idx].mean())

    # CV1 via 1D DE: minimise Fréchet distance of ATCS in Region I
    T_region_1L = T_data[:boundary_idx + 1]
    omega_region_1L = omega_data[:boundary_idx + 1]

    def cv1_objective(params):
        CV1 = params[0]
        T_mod = compute_T_1L(k1, omega_region_1L, CV1)
        return discrete_frechet(T_region_1L, T_mod)

    res = differential_evolution(cv1_objective, bounds=[(1.5e6, 5e6)],
                                 strategy='rand1bin',
                                 maxiter=500, tol=1e-8, seed=42)
    CV1 = res.x[0]

    print(f"[Stage 1] k1 = {k1:.6f} W/(m·K),  CV1 = {CV1:.4e} J/(m³·K)")
    print(f"          Region I: indices 0 .. {boundary_idx}  "
          f"(ω = {float(omega_data[0]):.2e} .. {float(omega_data[boundary_idx]):.2e})")

    return k1, CV1, boundary_idx

# ════════════════════════════════════════════════════════════════
# Universal adaptive frequency sweep  (Stages 2, 3, ...)
# ════════════════════════════════════════════════════════════════

def adaptive_sweep(T_data, omega_data, loss_fn_maker, bounds,
                   anchor_idx, max_idx,
                   sweep_step=3, data_stride=3,
                   s0=S0, z_tau=Z_TAU,
                   de_popsize=8, de_tol=0.02,
                   max_points=30,
                   min_end_override=None):
    loss_history = []
    param_history = []
    sweep_endpoints = []  # data index at each sweep step
    prev_x = None  # for warm-starting DE

    WARMUP_STEPS = s0

    if min_end_override is not None:
        min_end = min_end_override
    else:
        min_end = anchor_idx + sweep_step * 3

    for curr_end in range(min_end, max_idx, sweep_step):
        T_slice = T_data[anchor_idx:curr_end:data_stride]
        omega_slice = omega_data[anchor_idx:curr_end:data_stride]

        if len(T_slice) < 10:
            continue

        # ── Cap frequency points to keep DE cost constant ─────
        if max_points and len(T_slice) > max_points:
            indices = np.linspace(0, len(T_slice) - 1, max_points, dtype=int)
            T_slice = T_slice[indices]
            omega_slice = omega_slice[indices]

        loss_fn = loss_fn_maker(T_slice, omega_slice)

        # Warm-start: seed DE with previous step's best solution
        x0 = prev_x if prev_x is not None else None
        res = differential_evolution(loss_fn, bounds,
                                     strategy='rand1bin',
                                     popsize=de_popsize, tol=de_tol,
                                     maxiter=DE_MAXITER, x0=x0)
        prev_x = res.x.copy()

        loss_history.append(res.fun)
        param_history.append(res.x.copy())
        sweep_endpoints.append(curr_end)

        # Print progress
        step_idx = len(loss_history) - 1
        print(f"  sweep step {step_idx:3d}  idx={curr_end:4d}  "
              f"loss={res.fun:.4e}  params={res.x}")

        # Check for mutation (skip warmup steps where loss is unreliable)
        if step_idx >= WARMUP_STEPS and \
           detect_mutation(res.fun, loss_history[:-1],
                           s0=s0, z_tau=z_tau, bilateral=False):
            # ── Boundary truncation ────────────────────────────
            # Roll back to the MIDPOINT of the rolling window.
            # The s0 recent steps form the baseline; the middle
            # element is the last step still safely within the
            # plateau, while the later half may already contain
            # early leakage from the next layer.
            safe_i = max(0, step_idx - s0 // 2)
            best_params = param_history[safe_i]
            boundary_idx = sweep_endpoints[safe_i]

            print(f"  * Mutation at step {step_idx} (idx {curr_end}).  "
                  f"Safe boundary -> step {safe_i} (idx {boundary_idx}).")
            return best_params, boundary_idx, True, loss_history, param_history

    # ── Natural termination ────────────────────────────────────
    # Sweep reached ω_min without mutation → current model order is sufficient.
    best_params = param_history[-1]
    boundary_idx = sweep_endpoints[-1]
    print(f"  ✦ No mutation. Natural termination at idx {boundary_idx}.")
    return best_params, boundary_idx, False, loss_history, param_history


# ════════════════════════════════════════════════════════════════
# Full inversion pipeline
# ════════════════════════════════════════════════════════════════

def invert_case(T_data, depth_data,
                bounds_stage2, bounds_stage3,
                sweep_step=4, data_stride=4):
    omega_data = D_OMEGA / depth_data.pow(2)

    # ── Stage 1 ─────────────────────────────────────────────────
    print("=" * 60)
    print("STAGE 1 — Superficial layer extraction (ATCS plateau)")
    print("=" * 60)
    k1, CV1, s1_bnd = stage1(T_data, omega_data)

    # ── Stage 2 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2 — Second-layer identification (adaptive sweep)")
    print("=" * 60)

    def make_loss_2L(T_slice, omega_slice):
        return partial(loss_2L,
                       T_exp=T_slice, omega_vec=omega_slice,
                       k1=k1, CV1=CV1)

    params2, s2_bnd, mut2, losses2, phist2 = adaptive_sweep(
        T_data, omega_data,
        loss_fn_maker=make_loss_2L,
        bounds=bounds_stage2,
        anchor_idx=s1_bnd,
        max_idx=len(T_data),
        sweep_step=sweep_step,
        data_stride=data_stride,
    )
    L1, k2, CV2 = params2
    print(f"\n  → L1 = {L1:.4e} m,  k2 = {k2:.6f} W/(m·K),  CV2 = {CV2:.4e} J/(m³·K)")

    # ── Stage 3 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 3 — Third-layer identification (single DE, full window)")
    print("=" * 60)

    # Use the window from s2_bnd (Stage 2 safe boundary) to the last frequency point.
    # Stage 3 uses stride=5 for sampling.
    T_slice_3 = T_data[s2_bnd::5]
    omega_slice_3 = omega_data[s2_bnd::5]

    loss_fn_3L = partial(loss_3L,
                         T_exp=T_slice_3, omega_vec=omega_slice_3,
                         k1=k1, CV1=CV1, L1=L1, k2=k2, CV2=CV2)

    res3 = differential_evolution(loss_fn_3L, bounds=bounds_stage3,
                                  strategy='rand1bin',
                                  popsize=DE_POPSIZE, tol=DE_TOL,
                                  maxiter=DE_MAXITER,
                                  mutation=DE_F, recombination=DE_CR,
                                  seed=42)
    params3 = res3.x
    L2, k3, CV3 = params3
    s3_bnd = len(T_data) - 1
    mut3 = False
    losses3 = [res3.fun]

    print(f"  d_F* = {res3.fun:.4e}")
    print(f"\n  → L2 = {L2:.4e} m,  k3 = {k3:.6f} W/(m·K),  CV3 = {CV3:.4e} J/(m³·K)")
    print("\n  ✓  Single-DE termination — 3-layer model covers full spectrum.")

    return {
        "k1": k1, "CV1": CV1,
        "L1": L1, "k2": k2, "CV2": CV2,
        "L2": L2, "k3": k3, "CV3": CV3,
        "s1_boundary": s1_bnd,
        "s2_boundary": s2_bnd,
        "s3_boundary": s3_bnd,
        "stage2_mutation": mut2,
        "stage3_mutation": mut3,
        "losses_stage2": losses2,
        "losses_stage3": losses3,
    }


# ════════════════════════════════════════════════════════════════
# Data I/O
# ════════════════════════════════════════════════════════════════

def read_cases_from_excel(df, identifier: str):
    """Read a single case (T, depth) from the Excel data format."""
    first_row = df.iloc[0]
    matches = first_row[first_row == identifier]
    if matches.empty:
        raise ValueError(f"Identifier '{identifier}' not found in header row.")
    col = matches.index[0]
    block = df.iloc[1:, col + 1: col + 3]

    def col_to_tensor(n):
        arr = block.iloc[:, n].dropna().to_numpy(dtype=float)
        return torch.tensor(arr, dtype=torch.float32)

    return col_to_tensor(0), col_to_tensor(1)  # T, depth


# ════════════════════════════════════════════════════════════════
# Batch runner
# ════════════════════════════════════════════════════════════════

def run_batch(filepath, identifiers,
              bounds_stage2=None, bounds_stage3=None,
              sweep_step=4, data_stride=4,
              output_path=None):
    if bounds_stage2 is None:
        bounds_stage2 = [(30e-6, 400e-6), (0.18, 0.50), (1.8e6, 3.6e6)]
    if bounds_stage3 is None:
        bounds_stage3 = [(150e-6, 1400e-6), (0.30, 0.55), (2.5e6, 4.5e6)]

    if output_path is None:
        output_path = os.path.join(os.path.dirname(filepath), "results_pipeline.xlsx")

    df = pd.read_excel(filepath, header=None)
    all_results = {}

    # Load existing results if the output file already exists,
    # so we can resume an interrupted batch run.
    if os.path.exists(output_path):
        try:
            existing = pd.read_excel(output_path, index_col=0)
            all_results = existing.T.to_dict()
            print(f"Loaded {len(all_results)} existing results from {output_path}")
        except Exception:
            pass

    for i, ident in enumerate(identifiers):
        if ident in all_results:
            print(f"\n[{i+1}/{len(identifiers)}] {ident} — already computed, skipping.")
            continue

        print(f"\n{'#' * 60}")
        print(f"#  [{i+1}/{len(identifiers)}] Case: {ident}")
        print(f"{'#' * 60}")

        try:
            import time as _time
            _t0 = _time.perf_counter()

            T1, depth1 = read_cases_from_excel(df, ident)
            result = invert_case(T1, depth1,
                                 bounds_stage2=bounds_stage2,
                                 bounds_stage3=bounds_stage3,
                                 sweep_step=sweep_step,
                                 data_stride=data_stride)

            elapsed = _time.perf_counter() - _t0
            print(f"\n  ⏱  Case {ident} finished in {elapsed:.1f} s")

            # Store numerical results (exclude lists)
            all_results[ident] = {
                k: v for k, v in result.items()
                if isinstance(v, (int, float, bool, np.floating))
            }
            all_results[ident]["elapsed_s"] = round(elapsed, 2)
        except Exception as e:
            print(f"\n  !! ERROR on {ident}: {e}")
            all_results[ident] = {"error": str(e)}

        # ── Incremental save after every case ──────────────────
        results_df = pd.DataFrame(all_results).T
        results_df.to_excel(output_path)
        print(f"  [saved {len(all_results)}/{len(identifiers)} cases to {output_path}]")

    print(f"\nBatch complete. {len(all_results)} results in {output_path}")
    return all_results


# ════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    filepath = "torchdata_60cases.xlsx"
    identifiers = [
"50-250-0", "50-500-0", "50-750-0", "50-1000-0",
"100-250-0", "100-500-0", "100-750-0", "100-1000-0",
"150-250-0", "150-500-0", "150-750-0", "150-1000-0",
"200-250-0", "200-500-0", "200-750-0", "200-1000-0",
"250-250-0", "250-500-0", "250-750-0", "250-1000-0",
"50-250-50", "50-500-50", "50-750-50", "50-1000-50",
"100-250-50", "100-500-50", "100-750-50", "100-1000-50",
"150-250-50", "150-500-50", "150-750-50", "150-1000-50",
"200-250-50", "200-500-50", "200-750-50", "200-1000-50",
"250-250-50", "250-500-50", "250-750-50", "250-1000-50",
"50-250-75", "50-500-75", "50-750-75", "50-1000-75",
"100-250-75", "100-500-75", "100-750-75", "100-1000-75",
"150-250-75", "150-500-75", "150-750-75", "150-1000-75",
"200-250-75", "200-500-75", "200-750-75", "200-1000-75",
"250-250-75", "250-500-75", "250-750-75", "250-1000-75"
        ]

    bounds_s2 = [(30e-6, 350e-6), (0.18, 0.60), (1.8e6, 4.7e6)]
    bounds_s3 = [(150e-6, 1400e-6), (0.18, 0.6), (1.8e6, 4.7e6)]

    run_batch(filepath, identifiers,
              bounds_stage2=bounds_s2,
              bounds_stage3=bounds_s3,
              sweep_step=4, data_stride=4)
