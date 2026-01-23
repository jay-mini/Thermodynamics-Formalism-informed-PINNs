# Thermodynamic-Structure-Informed Physics-Informed Neural Networks

## Overview

Physics-informed neural networks (PINNs) offer a flexible framework for solving both forward and inverse problems of differential equations by embedding governing laws as soft constraints within the loss function.
Despite their success, standard PINNs often suffer from limited generalization, instability in inverse problems, and violations of fundamental physical principles, particularly in noisy or data-scarce regimes. 
This project presents a systematic comparative study of thermodynamic structure–informed PINNs, where different physical formulations are explicitly embedded into the PINN framework.
We investigate how Newtonian, Lagrangian, and Hamiltonian mechanics for conservative systems, and the Onsager variational principle (OVP) and Extended Irreversible Thermodynamics (EIT) for dissipative systems,
affect accuracy, robustness, physical consistency, and interpretability when solving forward and inverse problems. The goal is not to propose yet another PINN variant,
\but to provide quantitative guidance on how the choice of physical formulation shapes the loss landscape and the learned solution space.

## Key Advantages and Findings

### Conservative Systems

- **Newtonian PINNs (NM-PINNs)** accurately reconstruct trajectories but are insufficient for learning intrinsic physical quantities such as Lagrangians, Hamiltonians, and phase-space structures.
- **Lagrangian-mechanics PINNs (LM-PINNs)** show superior performance in parameter identification and robustness to noise, benefiting from variational structure.
- **Hamiltonian-mechanics PINNs (HM-PINNs)** explicitly preserve energy invariants and symplectic structure, yielding the most faithful phase-space reconstructions, though with reduced inverse-problem stability under strong noise.

### Dissipative Systems

- **NM-PINNs** can recover state variables but exhibit systematic bias when learning thermodynamic quantities such as Rayleighians, entropy functions, and entropy production.
- **Onsager variational principle PINNs (OVP-PINNs)** significantly improve stability and robustness in Rayleighian learning and parameter identification, especially in noisy settings.
- **Extended Irreversible Thermodynamics PINNs (EIT-PINNs)**, by enforcing entropy balance and entropy production, uniquely enable accurate recovery of entropy functions and fluxes, and outperform OVP-PINNs in thermodynamic consistency and interpretability while maintaining comparable accuracy.

Overall, the results demonstrate that explicitly embedding structure-preserving thermodynamic formulations is essential for physically consistent and interpretable PINNs, particularly for inverse problems.

## Benchmark Problems
The framework is evaluated on a broad set of canonical systems:
- **Conservative ODEs**: ideal mass–spring oscillator, simple pendulum, double pendulum
- **Dissipative ODEs**: damped pendulum
- **Dissipative PDEs**: diffusion equation, Fisher-KPP equation
Both forward and inverse problems are considered under clean and noisy data regimes.




