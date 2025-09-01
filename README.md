# Points of Interest

1. \( S^2_{\text{max}} = 1.95213 \times 10^{-8} \) at \( r = 33.0571 \)  
   → This is the radius of interest. At this point the analytical \( S \) makes a jump.  

2. \( f_{\text{max}} = 1.14112 \times 10^{-23} \) at \( r = 0.121059 \)

---

# Notes for Myself

1. Make an analogy with **Lotka–Volterra dynamics**.  
2. What is special about \( r = 33.0571 \)?  
3. What is \( r = 0.121059 \)?  
4. Why is \( f \) not exactly zero?

---

# Equations from the Python Code

### Safe denominators

\[
g = r^{2} B
\]

\[
g_{\text{safe}} =
\begin{cases}
  \text{sign}(g + \varepsilon_{\text{nonan}})\,\varepsilon_B, & |g| < \varepsilon_B, \\
  g, & \text{otherwise}
\end{cases}
\]

\[
B_{\text{safe}} =
\begin{cases}
  \text{sign}(B + \varepsilon_{\text{nonan}})\,\varepsilon_B, & |B| < \varepsilon_B, \\
  B, & \text{otherwise}
\end{cases}
\]

---

### Definition of \( S' \)

\[
S' = \frac{y}{g_{\text{safe}}}
\]

---

### Curvature-like scalar

\[
R = \frac{2(w-1)}{r^{2}} - 12\kappa
\]

---

### From the \(S\)-equation

\[
y' = r^{2}\left(\lambda S^{2} - \left(\frac{\omega^{2}}{B_{\text{safe}}} - \frac{R}{6}\right)\right) S
\]

---

### Forced derivative of \(B\)

\[
B' = \frac{2m}{r^{2}} + \gamma - 2\kappa r
\]

---

### Second derivative of \(S\)

Since

\[
S' = \frac{y}{r^{2} B}, \qquad g' = 2rB + r^{2}B',
\]

we obtain

\[
S'' = \frac{y'}{g_{\text{safe}}} - S' \cdot \frac{g'}{g_{\text{safe}}}
\]

---

### Definition of \(f\)

\[
f = -\frac{1}{2\alpha}\left((S')^{2} + S S''\right)
\]

---

### Metric ODEs

\[
\begin{aligned}
w'     &= \tfrac{1}{2} r^{3} f, \\
m'     &= \tfrac{1}{12} r^{4} f, \\
\gamma'&= -\tfrac{1}{2} r^{2} f, \\
\kappa'&= -\tfrac{1}{6} r f
\end{aligned}
\]
