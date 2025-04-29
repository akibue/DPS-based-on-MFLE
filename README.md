# DPS-based-on-MFLE

Each folders contain the code and results for the numerical experiment found at https://arxiv.org/abs/2501.17394.

It includes six files:

1. **MCtest.py**: This Python code is used to solve the optimization problem under a separable cone. It utilizes the PICOS and QICS packages.
2. **theta.txt**: This file contains a parameter that describes the strength of entanglement of the resource state (or target state for the case of entanglement distillation.)
3. **PPT.txt**: This file contains upper bounds on the solution of the optimization problem for each parameter obtained by the PPT relaxation.
4. **DPS2nd.txt**: This file contains upper bounds on the solutions of the optimization problem for each parameter obtained by the second level of the DPS relaxation.
5. **MFLE.txt**: This file contains upper bounds on the solutions of the optimization problem for each parameter obtained by the PPT relaxation and the linear constraints resulting from the minimum finite linear extension (MFLE).
6. **lowerbound.txt**: This file contains lower bounds on the solutions of the optimization problem for each parameter using an epsilon-net approximation (or an analytical method for the case of entanglement distillation.)
