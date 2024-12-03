# Localized Vortices in Spinor Polariton Condensates with Spin-Orbit Coupling

This section examines the formation and stability of localized vortices in spinor polariton condensates, focusing on the effects of spin-orbit coupling. Quantized vortices in polariton condensates are characterized by phase singularities and topological charges, which give rise to quantized orbital angular momentum (OAM). These vortices have potential applications in information processing and qubit analogs, where their topological charge can serve as bit or qubit values.

Polariton condensates' spin degrees of freedom, originating from coupled exciton and photon polarization states, enable complex vortex states such as:
- **Half-quantum vortices**: A vortex in one spin component and no vortex in the other ($m_\pm \neq 0 \wedge m_\mp = 0$).
- **Full-spin vortices**: Parallel ($m_+ = m_-$) or antiparallel ($m_+ = -m_-$) vortices in both spin components.

Spin-orbit interaction, induced by TE-TM (longitudinal-transverse) splitting, couples the vortex states of the two spin components, breaking symmetry and creating diverse vortex configurations.
To describe the dynamics of the spinor polariton condensate under non-resonant excitation, we solve the following form of the Gross-Pitaevskii model with the phenomenological extension of an excitonic reservoir:

$$
\partial_t\psi_\pm= \frac{-i}{\hbar}\biggl(-\frac{\hbar^2}{2m_\mathrm{eff}}\nabla^2+g|\psi_\pm|^2+g_\mathrm{r}n_\pm +g_\mathrm{x}n_\mp + \frac{i\hbar}{2} [Rn-\gamma] \biggr) \psi_\pm +J_\pm \psi_∓ ,
$$
$$
\partial_t n_\pm = P_\pm(x,y) - (\gamma_\mathrm{r}+R|\psi_\pm|^2)n_\pm.
$$

Here $m_\mathrm{eff}= 10^{-4}m_\mathrm{e}$ defines the effective polariton mass, $\gamma=0.15~\mathrm{ps^{-1}}$ the polariton loss rate and $\gamma_\mathrm{r}=1.5\gamma$ the reservoir loss rate, $g=3~\mathrm{\upmu eV\upmu m^2}$ the polariton-polariton and $g_\mathrm{r}=2g$ the polariton-exciton interaction strength for particles of identical spin, $g_\mathrm{x}=0.2g$ the cross-interaction strength between spinor components of the condensate and $R = 0.01 \mathrm{ps^{-1}\upmu m^2}$ the condensation rate. The TE-TM splitting operator is modeled by the operator $J_\pm= \Delta_\mathrm{LT}(\partial_x ∓ i\partial_y)^2$ and the TE-TM splitting strength is defined by $\Delta_\mathrm{LT}=0.025~\mathrm{meV\upmu m^2}$.

A $x$-linearly polarized continuous-wave pump with a ring profile was used:

$$
P_\pm(\textbf{r}) = P_0\frac{\textbf{r}^2}{w_p^2} \exp\left(\frac{\textbf{r}^2}{w_p^2}\right),
$$

with $P_0 = 100~\mu\text{m}^{-2}\text{ps}^{-1}$ and radius $w_p$.

High computational efficiency allowed systematic scanning of vortex states across various topological charge configurations ($m_+ - m_-$, $m_+ + m_-$), as shown in Fig.~\ref{fig:example3}. These simulations revealed diverse vortex patterns and their associated real-space symmetries, driven by spin-orbit coupling.

Each simulation evaluated $10~\text{ns}$ of temporal evolution on a $500 \times 500$ grid, with runtime of approximately 1 minute per run using an NVIDIA RTX 4090 GPU and AMD EPYC 7443P CPU. 

![example3_overview_with_symmetries_tc.png](example3_overview_with_symmetries_tc.png)

The figure shows:  
- **Grid of vortex states** arranged by the topological charge difference ($m_+ - m_-$, vertical axis) and sum ($m_+ + m_-$, horizontal axis).  
- **Density and phase distributions** of the circular polarizations for each vortex state, depicted with respective color bars.  
- **Real-space symmetries** resulting from spin-orbit interaction.  
At the bottom, a schematic shows the layout of densities and phases, with pump radii and plot window sizes for marked regions.
