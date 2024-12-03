# Algorithm-Assisted Localization of Exceptional Points in Reservoir-Coupled Polariton Condensates

This section investigates the localization of exceptional points (EPs) in reservoir-coupled polariton condensates, leveraging PHOENIX's computational efficiency. Exceptional points are singularities in non-Hermitian systems where eigenvalues and eigenvectors coalesce, enabling unique phenomena like loss-induced transparency and unidirectional invisibility. These points, particularly in nonlinear systems, present challenges for precise localization due to the coupling of mode gain, loss, and energy through nonlinearities. To describe the dynamics of the polariton condensate under non-resonant excitation, we solve the following form of the Gross-Pitaevskii model with the phenomenological extension of an excitonic reservoir:

$$
\partial_t\psi= \frac{-i}{\hbar}\biggl(-\frac{\hbar^2}{2m}\nabla^2+V(x,y)+g|\psi|^2+g_\mathrm{r} n + \frac{i\hbar}{2} [Rn-\gamma] \biggr) \psi,
$$
$$
\partial_t n = P(x,y) - (\gamma_\mathrm{r}+R|\psi|^2)n.
$$

Here $m=0.5\cdot 10^{-4}m_\mathrm{e}$ defines the effective polariton mass, $\gamma=0.16~\mathrm{ps^{-1}}$ the polariton loss rate and $\gamma_\mathrm{r}=1.5\gamma$ the reservoir loss rate, $g=6~\mathrm{\upmu eV\upmu m^2}$ the polariton-polariton and $g_\mathrm{r}=2g$ the polariton-exciton interaction strength, and $R = 0.01 \mathrm{ps^{-1}\upmu m^2}$ the condensation rate.

A second-order exceptional point is studied in a double-well potential described by:

$$
V(x, y) = V_1 f(x_1, w_v) + V_2 f(x_2, w_v),
$$

where the envelope function is:

$$
f(x_\text{shift}, w) = \exp\left(-\frac{(x-x_\text{shift})^2 + y^2}{w^2}\right)^2.
$$

The well depths are $V_1 = -2.2~\text{meV}$ and $V_2 = -2.0~\text{meV}$, with a separation of $d = 4~\mu\text{m}$ and a well radius $w_v = 1.5~\mu\text{m}$. Nonresonant excitation is applied via a Gaussian-shaped pump:

$$
P(x, y) = P_1 f(x_1, w_p) + P_2 f(x_2, w_p),
$$

where $w_p = 1~\mu\text{m}$ and pump intensities are $P_1 = 12~\text{ps}^{-1}\mu\text{m}^{-2}$ (left) and $P_2$ (right, variable). 

By gradually increasing $P_2$, the exceptional point is traced, showcasing condensation dynamics, mode energy bifurcation, and the characteristic condensation switch-off near the EP. The EP is automatically localized using MATLAB’s one-dimensional optimizer on the interval $P_2 = [0, 20]~\text{ps}^{-1}\mu\text{m}^{-2}$. Iterations converge to the EP, minimizing the integrated density in the system, a process completed in approximately one hour with GPU acceleration.

![example2_overview_tc2.png](example2_overview_tc2.png)

The figure shows:  
(a) Spatial distribution of the Gaussian pump intensity in $\text{ps}^{-1}\mu\text{m}^{-2}$.  
(b) Double-well potential distribution in $\text{meV}$.  
(c) Integrated density of polariton modes and mode energy bifurcation at the exceptional point. Insets show bonding and antibonding modes localized in the wells.  
(d) Algorithm-assisted localization of the exceptional point with increasing iterations marked by progressively lighter colors.
