#!/usr/bin/env python3
import adcc

from pyscf import gto, scf
from relaxation import relax_cvs

from matplotlib import pyplot as plt

# Run SCF in pyscf
mol = gto.M(
    atom="""
        O 0 0 0
        H 0 0 1.795239827225189
        H 1.693194615993441 0 -0.599043184453037
    """,
    basis='cc-pcvtz',
    unit="Bohr"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-12
scfres.conv_tol_grad = 1e-9
scfres.kernel()

# Run CVS-ADC(2) and relax to ADC(2) for O-edge
state = adcc.cvs_adc2(scfres, conv_tol=1e-8, core_orbitals=1, n_singlets=10)
state_relaxed = relax_cvs(scfres, "adc2", state)

# Print results
print()
print(state.describe())
print()
print()
print(state_relaxed.describe())
print()

state.plot_spectrum(label="CVS-ADC(2)")
state_relaxed.plot_spectrum(label="General ADC(2)")
print("Residual norms: ", state_relaxed.residual_norms)

plt.legend()
plt.savefig("water_relaxed.pdf")
