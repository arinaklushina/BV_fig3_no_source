# bvp_packaged.py
# -----------------------------------------------------------
# Solves the coupled system for Y = [S, y, w, m, gamma, kappa]
# with algebraic:
#   B   = w - 2*m/r + gamma*r + kappa*r**2
#   S'  = y / (r**2 * B)
# and ODEs:
#   dy = r**2 * (lam*S**2 - (omega**2/B - R/6)) * S
#   R  = 2*(w - 1)/r**2 - 12*kappa
#   dB = 2*m/r**2 + gamma - 2*kappa*r            <-- forced per request
#   f  = -(1/(2*alpha)) * ((S')**2 + S*S'')
#   dw      =  0.5       * r**3 * f
#   dm      = (1.0/12.0) * r**4 * f
#   dgamma  = -0.5       * r**2 * f
#   dkappa  = -(1.0/6.0) * r    * f
#
# BCs (independent, consistent):
#   Left:  y(0)=0, w(0)=1, gamma(0)=0, m(0)=0
#   Right: S(∞)=0, B''(∞)=0   (implemented as 2*kappa(∞)=0)
#
# Saves plots:
#   1) B, B'                 -> B_Bp_ver2.png
#   2) S, S'                 -> S_Sp_ver2.png
#   3) w, m, gamma, kappa    -> MK_param_ver2.png
#   4) log–log (positive)    -> B_Bp_loglog_ver2.png
#   5) R(r)                  -> R_ver2.png
# -----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# ----- constants you can tune -----
OMEGA =0    # omega
LAM   = 1.0     # lambda
ALPHA = -1.0     # alpha
SIGMA = 1.0

R_MIN = 1e-3    # left boundary (>0)
R_MAX = 80.0    # "infinity"
NPTS  = 800     # initial mesh points

KAPPA_INF = 0.1 # only used for initial guess; not a BC

# small epsilons for robust divisions
EPS_B     = 1e-12     # for divides by B or r^2 B
EPS_NONAN = 1e-30


def rhs(r, Y, omega, lam, alpha):
    """
    Right-hand side dY/dr for the first-order system.

    Y rows (each is shape (m,)):
        Y[0] = S
        Y[1] = y         (y = r^2 * B * S')
        Y[2] = w
        Y[3] = m
        Y[4] = gamma
        Y[5] = kappa
    """
    r = np.asarray(r)
    S, y, w, m, gamma, kappa = Y

    # B from metric functions
    B = w - 2.0*m/r + gamma*r + kappa*r**2

    # safe denominators
    g      = r**2 * B
    g_safe = np.where(np.abs(g) < EPS_B, np.sign(g + EPS_NONAN) * EPS_B, g)
    B_safe = np.where(np.abs(B) < EPS_B, np.sign(B + EPS_NONAN) * EPS_B, B)

    # S' from definition
    dS = y / g_safe

    # curvature-like scalar
    R = 2.0*(w - 1.0)/r**2 - 12.0*kappa

    # y' from S-equation
    dy = r**2 * (lam*S**2 - (omega**2/B_safe - R/6.0)) * S

    # ----- dB forced per request -----
    dB = 2.0*m/r**2 + gamma - 2.0*kappa*r

    # S'' from S' = y / (r^2 B): S'' = (y'/g) - S' * (g'/g), g' = 2 r B + r^2 B'
    gprime = 2.0*r*B + r**2 * dB
    ddS = (dy / g_safe) - dS * (gprime / g_safe)

    # f = -(1/(2 alpha)) * ((S')**2 + S S'')
    f = -(1.0/(2.0*alpha)) * (dS**2 + S*ddS)

    # metric ODEs
    dw     =  0.5        * r**3 * f
    dm     = (1.0/12.0)  * r**4 * f
    dgamma = -0.5        * r**2 * f
    dkappa = -(1.0/6.0)  * r    * f

    dY = np.vstack((dS, dy, dw, dm, dgamma, dkappa))
    return dY


def bc(ya, yb):
    """
    Independent boundary conditions:
      Left (r ~ 0):   y=0, w=1, gamma=0, m=0
      Right (r ~ ∞):  S=0, B''(∞)=0  → implemented as 2*kappa(∞)=0
    """
    Sa, ya_var, wa, ma, ga, ka = ya
    Sb, yb_var, wb, mb, gb, kb = yb
    return np.array([
        ya_var - 0.0,        # y(0) = 0  ⇒ S'(0)=0
        wa     - 1.0,        # w(0) = 1
        ga     - 0.0,        # gamma(0) = 0
        ma     - 0.0,        # m(0) = 0
        Sb     - 0.0,        # S(∞) = 0
        2.0*kb - 0.0         # B''(∞) = 0 (asymptotically, here → -2*kappa ≈ 0 as well)
    ])


def initial_guess(r):
    """Reasonable initial profiles that roughly satisfy far/near behavior."""
    S0 = np.exp(-r / 5.0)

    w0 = np.ones_like(r)
    m0 = np.zeros_like(r)
    g0 = np.zeros_like(r)     # gamma
    k0 = KAPPA_INF * np.ones_like(r)  # just a guess; not enforced as BC

    B0  = w0 - 2.0*m0/r + g0*r + k0*r**2
    S0p = -(1.0/5.0) * np.exp(-r / 5.0)
    y0  = r**2 * B0 * S0p     # y = r^2 B S'

    return np.vstack([S0, y0, w0, m0, g0, k0])


def _compute_B_and_Bp(r, S, y, w, m, gamma, kappa):
    """Helper to compute B and B' on any sampling of r."""
    B  = w - 2.0*m/r + gamma*r + kappa*r**2
    g  = r**2 * B
    g  = np.where(np.abs(g) < EPS_B, np.sign(g + EPS_NONAN) * EPS_B, g)
    Sp = y / g
    # ----- B' forced per request -----
    Bp = 2.0*m/r**2 + gamma - 2.0*kappa*r
    return B, Bp, Sp


def main():
    # mesh
    r = np.linspace(R_MIN, R_MAX, NPTS)

    # initial guess
    Y0 = initial_guess(r)

    # wrap rhs with constants
    def fun(r_, Y_):
        return rhs(r_, Y_, OMEGA, LAM, ALPHA)

    # solve
    sol = solve_bvp(fun, bc, r, Y0, tol=1e-6, max_nodes=200000, verbose=2)

    print("\n=== BVP status ===")
    print(f"Converged: {sol.success} | message: {sol.message}")
    print("BC residuals (abs):", np.abs(bc(sol.y[:, 0], sol.y[:, -1])))

    # dense sampling for plots
    rr       = np.linspace(R_MIN, R_MAX, 2000)
    rr_short = np.linspace(R_MIN, 0.1,   1000)  # zoom near the origin

    Y        = sol.sol(rr)        # shape (6, len(rr))
    Ys       = sol.sol(rr_short)  # shape (6, len(rr_short))

    S, y, w, m, gamma, kappa   = Y
    Ss, ys, ws, ms, gs, ks     = Ys

    # Build B, B' (on both grids), and S'
    B,  Bp,  Sp  = _compute_B_and_Bp(rr,       S,  y,  w,  m,  gamma,  kappa)
    Bs, Bps, Sps = _compute_B_and_Bp(rr_short, Ss, ys, ws, ms, gs, ks)

    # curvature for plotting R(r)
    Rcurv = 2.0*(w - 1.0)/rr**2 - 12.0*kappa

    # ---------- plots (and SAVE) ----------
    plt.figure()
    plt.plot(rr, B,  label="B(r)")
    plt.plot(rr, Bp, label="B'(r)", linestyle="--")
    plt.xlabel("r"); plt.ylabel("value"); plt.title("B and B'")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("B_Bp_ver2.png", dpi=300, bbox_inches="tight")



    plt.figure()
    plt.plot(rr, S,  label="S(r)")
    plt.plot(rr, Sp, label="S'(r)", linestyle="--")
    plt.xlabel("r"); plt.ylabel("value"); plt.title("S and S'")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("S_Sp_ver2.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(rr, w,     label="w(r)")
    plt.plot(rr, m,     label="m(r)")
    plt.plot(rr, gamma, label=r"$\gamma(r)$")
    plt.plot(rr, kappa, label=r"$\kappa(r)$")
    plt.xlabel("r"); plt.ylabel("value"); plt.title("w, m, gamma, kappa")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("MK_param_ver2.png", dpi=300, bbox_inches="tight")

    # log–log positive segments
    B_pos  = np.where(B  > 0, B,  np.nan)
    Bp_pos = np.where(Bp > 0, Bp, np.nan)
    plt.figure()
    plt.loglog(rr, B_pos,  label="B(r) > 0")
    plt.loglog(rr, Bp_pos, label="B'(r) > 0", linestyle="--")
    plt.xlabel("r"); plt.ylabel("value")
    plt.title("Log–log: B and B' (positive segments)")
    plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("B_Bp_loglog_ver2.png", dpi=300, bbox_inches="tight")

    # R(r)
    plt.figure()
    plt.plot(rr, Rcurv, label="R(r)")
    plt.xlabel("r"); plt.ylabel("R")
    plt.title("R(r)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("R_ver2.png", dpi=300, bbox_inches="tight")

    # ===== Compare S1 (numerical) with S2(r) = S0 * a / (r + a) =====
    EPS_A = 1e-10
    S0_val = S[0]  # value at r_min

    # a(r) from both formulas
    a1_arr = np.where(np.abs(gamma) > EPS_A, (w + 1.0) / gamma, np.nan)
    a2_arr = np.where(np.abs(1.0 - w) > EPS_A, 6.0 * m / (1.0 - w), np.nan)

    # robust constants (nan-median over valid points)
    def nanmedian_or_nan(x):
        x = x[np.isfinite(x)]
        return np.nan if x.size == 0 else np.nanmedian(x)

    a1_const = nanmedian_or_nan(a1_arr)
    a2_const = nanmedian_or_nan(a2_arr)

    # analytic curves (only if a is finite and not ~0)
    S2_a1 = None if not np.isfinite(a1_const) or np.abs(a1_const) < EPS_A else S0_val * a1_const / (rr + a1_const)
    S2_a2 = None if not np.isfinite(a2_const) or np.abs(a2_const) < EPS_A else S0_val * a2_const / (rr + a2_const)

    plt.figure()
    plt.plot(rr, S, label="S1 (numerical)")
    if S2_a1 is not None:
        plt.plot(rr, S2_a1, "--", label=f"S2, a=(w+1)/γ ≈ {a1_const:.3g}")
    if S2_a2 is not None:
        plt.plot(rr, S2_a2, ":", label=f"S2, a=6m/(1−w) ≈ {a2_const:.3g}")
    plt.xlabel("r");
    plt.ylabel("S")
    plt.title("S1 vs S2(r) = S0·a/(r+a)")
    plt.grid(True, alpha=0.3);
    plt.legend();
    plt.tight_layout()
    plt.savefig("S_compare.png", dpi=300, bbox_inches="tight")

    # --- S(y) with r as parameter (color = r) ---
    valid = np.isfinite(y) & np.isfinite(S)
    plt.figure()
    sc = plt.scatter(y[valid], S[valid], c=rr[valid], s=12)
    plt.xlabel("y");
    plt.ylabel("S")
    plt.title("S(y) (points colored by r)")
    plt.colorbar(sc, label="r")
    plt.grid(True, alpha=0.3);
    plt.tight_layout()
    plt.savefig("S_vs_y_param_r_num.png", dpi=300, bbox_inches="tight")

    # ---- Find max of S2 from the left (for a = (w+1)/gamma and a = 6 m / (1-w)) ----
    EPS_A = 1e-10
    S0_val = S[0]

    a1_arr = np.where(np.abs(gamma) > EPS_A, (w + 1.0) / gamma, np.nan)
    a2_arr = np.where(np.abs(1.0 - w) > EPS_A, 6.0 * m / (1.0 - w), np.nan)

    def robust_const(a_arr):
        aa = a_arr[np.isfinite(a_arr)]
        return np.nan if aa.size == 0 else np.nanmedian(aa)

    a1 = robust_const(a1_arr)
    a2 = robust_const(a2_arr)

    def s2_curve(a_const):
        if not np.isfinite(a_const) or np.abs(a_const) < EPS_A:
            return None
        return S0_val * a_const / (rr + a_const)

    def find_max_from_left(S2):
        if S2 is None:
            return None
        valid = np.isfinite(S2)
        if not np.any(valid):
            return None
        # global max over the grid (i.e., scanning from left)
        idx = np.nanargmax(np.where(valid, S2, -np.inf))
        return idx, rr[idx], S2[idx]

    S2_a1 = s2_curve(a1)
    S2_a2 = s2_curve(a2)

    res1 = find_max_from_left(S2_a1)
    res2 = find_max_from_left(S2_a2)

    if res1:
        i1, r1, smax1 = res1
        print(f"S2 max (a=(w+1)/gamma ≈ {a1:.6g}):  S2_max = {smax1:.6g} at r = {r1:.6g}")
    if res2:
        i2, r2, smax2 = res2
        print(f"S2 max (a=6m/(1-w) ≈ {a2:.6g}):    S2_max = {smax2:.6g} at r = {r2:.6g}")

    # --- f(r) from S, S'', alpha ---
    B_safe = np.where(np.abs(B) < EPS_B, np.sign(B + EPS_NONAN) * EPS_B, B)
    dy_expr = rr ** 2 * (LAM * S ** 2 - (OMEGA ** 2 / B_safe - Rcurv / 6.0)) * S
    dB_forced = 2.0 * m / rr ** 2 + gamma - 2.0 * kappa * rr
    g = rr ** 2 * B_safe
    g_safe = np.where(np.abs(g) < EPS_B, np.sign(g + EPS_NONAN) * EPS_B, g)
    gprime = 2.0 * rr * B_safe + rr ** 2 * dB_forced
    ddS = (dy_expr / g_safe) - Sp * (gprime / g_safe)
    f_rr = -(1.0 / (2.0 * ALPHA)) * (Sp ** 2 + S * ddS)

    # --- find maximum of f(r) (left-to-right scan over rr) ---
    valid = np.isfinite(f_rr)
    idx_max = None
    if np.any(valid):
        idx_max = np.nanargmax(np.where(valid, f_rr, -np.inf))
        r_max = rr[idx_max]
        f_max = f_rr[idx_max]
        print(f"f_max = {f_max:.6g} at r = {r_max:.6g}")

    # --- plot and save with max marker + radius in legend ---
    plt.figure()
    plt.plot(rr, f_rr, label="f(r)")
    if idx_max is not None:
        plt.scatter([r_max], [f_max], s=40, zorder=5, label=f"max at r = {r_max:.4g}")
    plt.xlabel("r");
    plt.ylabel("f")
    plt.title("f(r)")
    plt.grid(True, alpha=0.3);
    plt.legend();
    plt.tight_layout()
    plt.savefig("f_ver2.png", dpi=300, bbox_inches="tight")

    plt.show()

    print("\nSaved figures:")
    print("  - B_Bp_ver2.png")
    print("  - S_Sp_ver2.png")
    print("  - MK_param_ver2.png")
    print("  - B_Bp_loglog_ver2.png")
    print("  - R_ver2.png")




if __name__ == "__main__":
    main()
