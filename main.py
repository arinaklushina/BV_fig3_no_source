import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

plt.rcParams.update({'font.size': 15})
plt.rcParams['font.family'] = 'serif'  # Couldn't find Times New Roman, next best thing.
plt.rcParams['text.usetex'] = True

# ----------------------------
# Problem parameters
# ----------------------------
B0 = 1.0   # B(R_MIN)
dB0 = 0.0  # B'(R_MIN)
dddB0 = 0.0  # B'''(R_MIN)
df0 = 0.0    # f'(R_MIN)
ALPHA = 1.0  # Poisson constant

R_MIN = 1e-6
R_MAX = 60.0
NPTS = 1000

# --- Choose computational mesh for sampling/plotting ---
USE_LOG_MESH = False
if USE_LOG_MESH:
    r = np.geomspace(R_MIN, R_MAX, NPTS)
else:
    r = np.linspace(R_MIN, R_MAX, NPTS)

# Target boundary conditions at the right boundary
ddB_target = 2 * 0.1  # Boson stars with kappa=0.1
f_target = 0.0

# Define required parameters here
OMEGA = 1.0
LAM = 1.0

# ----------------------------
# ODE system (for IVP)
# y = [B, dB, ddB, dddB, f, df]
# ----------------------------
def ode_rhs_ivp(r, y):
    B, dB, ddB, dddB, f, df = y

    # Ricci scalar
    R = 2 * (1 - B) / r**2 - 4 * dB / r - ddB

    # Poisson: B'''' + 4 B'''/r = -(ALPHA/B)[ ... ]
    poiss_coeff = -ALPHA / B
    poiss_brack = (
        2 * (OMEGA**2) * (f**2) / B
        + B * (df**2)
        - (LAM * f**4) / 2
        + (dB + 2 * B / r) * 2 * f * df / 4
        - R * (f**2) / 12
    )
    ddddB = poiss_coeff * poiss_brack - 4 * dddB / r

    # Scalar field second derivative
    dy = (r**2) * (LAM * (f**3) - (OMEGA**2 / B - R / 6) * f)
    ddf = (dy - 2 * r * B * df - (r**2) * dB * df) / ((r**2) * B)

    return np.array([dB, ddB, dddB, ddddB, df, ddf], dtype=float)

# ----------------------------
# Shooting on ddB(R_MIN) to match ddB(R_MAX) = ddB_target
# Left-end fixed:
#   B(R_MIN)=B0, B'(R_MIN)=dB0, B'''(R_MIN)=dddB0, f(R_MIN)=f_target, f'(R_MIN)=df0
# Unknown:
#   ddB(R_MIN) = X  (we solve for X)
# ----------------------------
def shoot_residual(ddB_left):
    y0 = np.array([B0, dB0, ddB_left, dddB0, f_target, df0], dtype=float)
    sol_ivp = solve_ivp(
        ode_rhs_ivp, (R_MIN, R_MAX), y0,
        method='Radau', rtol=1e-8, atol=1e-10
    )
    if not sol_ivp.success or not np.isfinite(sol_ivp.y[2, -1]):
        return np.nan  # let the bracketer/solver handle this
    ddB_end = sol_ivp.y[2, -1]
    return ddB_end - ddB_target

def find_bracket(fun, x0, step0=0.1, max_expand=30):
    """Expand symmetrically around x0 until fun(xL) and fun(xR) have opposite signs."""
    # pick a starting step relative to scale
    step = step0 * (1.0 + abs(x0))
    f0 = fun(x0)
    if np.isfinite(f0) and abs(f0) < 1e-12:
        return (x0 - step, x0 + step)

    xL, xR = x0, x0
    fL, fR = f0, f0
    for k in range(max_expand):
        step *= 2.0
        xL = x0 - step
        xR = x0 + step
        fL = fun(xL)
        fR = fun(xR)
        if np.isfinite(fL) and np.isfinite(fR) and fL * fR < 0:
            return (xL, xR)
    return None

# Try to find a bracketing interval around ddB_target
bracket = find_bracket(shoot_residual, ddB_target, step0=0.1, max_expand=30)

if bracket is not None:
    root = root_scalar(shoot_residual, bracket=bracket, method='brentq', xtol=1e-10, rtol=1e-10, maxiter=100)
else:
    # fallback: secant from two nearby guesses
    root = root_scalar(shoot_residual, x0=ddB_target, x1=ddB_target + 0.2, method='secant', xtol=1e-10, rtol=1e-10, maxiter=200)

if not root.converged:
    raise RuntimeError("Shooting failed to converge for ddB(R_MIN).")

ddB_left_opt = root.root
print(f"Optimized ddB(R_MIN) = {ddB_left_opt:.12g}")

# Integrate one more time with optimal left ddB and sample on chosen r-grid
y0 = np.array([B0, dB0, ddB_left_opt, dddB0, f_target, df0], dtype=float)
sol_ivp = solve_ivp(
    ode_rhs_ivp, (R_MIN, R_MAX), y0,
    method='Radau', t_eval=r, rtol=1e-8, atol=1e-10
)
if not sol_ivp.success:
    raise RuntimeError(f"Radau IVP solver failed: {sol_ivp.message}")

# Extract solution
r_vals  = sol_ivp.t
B_sol   = sol_ivp.y[0]
dB_sol  = sol_ivp.y[1]
ddB_sol = sol_ivp.y[2]
f_sol   = sol_ivp.y[4]
df_sol  = sol_ivp.y[5]

# ----------------------------
# Plot + asymptotic fit
# ----------------------------
fig, ax = plt.subplots(figsize=(12, 7.5))
ax.set_xlabel(r"$r$")

params = [r"$B$", r"$B'$", r"$2B''$", r"$f$", r"$8P_r$"]
colors = plt.cm.hsv(np.linspace(0, 1, len(params), endpoint=False))

# ---- Fit B(r) ≈ a/r + b + c r + d r^2 (asymptotics) ----
r_fit_min = 5.0
r_fit_max = r_vals.max()

mask = (r_vals >= r_fit_min) & (r_vals <= r_fit_max) & np.isfinite(B_sol)
X = np.column_stack((1.0 / r_vals[mask], np.ones(mask.sum()), r_vals[mask], r_vals[mask]**2))
y = B_sol[mask]
a, b, c, d = np.linalg.lstsq(X, y, rcond=None)[0]

B_fit = a / r_vals + b + c * r_vals + d * (r_vals**2)

ax.plot(r_vals, B_fit, linestyle='--', linewidth=2,
        label=fr"fit: ${a:.3g}/r + {b:.3g} + {c:.3g}\,r + {d:.3g}\,r^2$")

w_est     = b
m_est     = -a / 2.0
gamma_est = c
kappa_est = d
print(f"Fit coefficients: a={a:.6g}, b={b:.6g}, c={c:.6g}, d={d:.6g}")
print(f"MK estimates   : w≈{w_est:.6g}, m≈{m_est:.6g}, γ≈{gamma_est:.6g}, κ≈{kappa_est:.6g}")
print(f"Check: 2κ≈{2*kappa_est:.6g} vs ddB_target={ddB_target:.6g}")

# Fit quality
resid = y - X @ np.array([a, b, c, d])
rmse  = np.sqrt(np.mean(resid**2))
sst   = np.sum((y - y.mean())**2)
r2    = 1 - np.sum(resid**2) / sst
print(f"Fit window r∈[{r_fit_min}, {r_fit_max}]  RMSE={rmse:.3e}, R^2={r2:.6f}")

# Solution plots
ax.plot(r_vals, B_sol,     label=params[0], color=colors[0])
ax.plot(r_vals, dB_sol,    label=params[1], color=colors[1])
ax.plot(r_vals, 2 * ddB_sol, label=params[2], color=colors[2])
ax.plot(r_vals, f_sol,     label=params[3], color=colors[3])

# Calculating P_r (kept exactly as in your code)
P_r = (
    -(OMEGA**2) * (f_sol**2) / (2 * B_sol)
    - B_sol * (df_sol**2) / 2
    + LAM * (f_sol**4) / 4
    - (dB_sol + 4 * B_sol / r_vals) * (2 * f_sol * df_sol) / 12
    - (dB_sol / r_vals - (1 - B_sol / r_vals**2)) * (f_sol**2) / 6
)
ax.plot(r_vals, 8 * P_r, label=params[4], color=colors[4])

ax.legend()
ax.set_ylim(0, 3)
ax.set_xlim(0, 10)
ax.grid()
plt.ticklabel_format(style='plain')
plt.subplots_adjust(hspace=0.)

plt.savefig("BV_fig3.png", dpi=300, bbox_inches="tight")
# plt.savefig("BV_fig3.pdf", bbox_inches="tight")  # optional vector copy
plt.show()
