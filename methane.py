#!/usr/bin/env python3
import adcc

from pyscf import gto, scf
from relaxation import relax_cvs

from matplotlib import pyplot as plt

# Run SCF in pyscf
mol = gto.M(
    atom="""
        C       0.000000    0.000000    0.000000
        H       0.628244    0.628244    0.628244
        H      -0.628244   -0.628244    0.628244
        H      -0.628244    0.628244   -0.628244
        H       0.628244   -0.628244   -0.628244
    """,
    basis='6-311++G**',
    unit="ang"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-12
scfres.conv_tol_grad = 1e-9
scfres.kernel()

# Run CVS-ADC(2)-x and relax to ADC(2)-x for C-edge
state = adcc.cvs_adc2x(scfres, conv_tol=1e-8, core_orbitals=1, n_singlets=6)
state_relaxed = relax_cvs(scfres, "adc2x", state)

# Print results
print()
print(state.describe())
print()
print()
print(state_relaxed.describe())
print()

state.plot_spectrum(label="orig CVS-ADC(2)-x")
state_relaxed.plot_spectrum(label="relaxed to ADC(2)-x")
print("Residual norms: ", state_relaxed.residual_norms)

plt.legend()
plt.savefig("methane_relaxed.pdf")
