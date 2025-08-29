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

    plt.show()

    print("\nSaved figures:")
    print("  - B_Bp_ver2.png")
    print("  - S_Sp_ver2.png")
    print("  - MK_param_ver2.png")
    print("  - B_Bp_loglog_ver2.png")
    print("  - R_ver2.png")


if __name__ == "__main__":
    main()
