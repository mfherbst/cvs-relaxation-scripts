#!/usr/bin/env python3
import adcc

from adcc import AdcBlockView, AdcMatrix, ReferenceState
from pyscf import gto, scf
from relaxation import relax_cvs

# Run SCF in pyscf
mol = gto.M(
    atom="""
        O 0 0 0
        H 0 0 1.795239827225189
        H 1.693194615993441 0 -0.599043184453037
    """,
    basis='cc-pcvdz',
    unit="Bohr"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-12
scfres.conv_tol_grad = 1e-9
scfres.kernel()

# Run CVS-ADC(3) singles and relax to ADC(3) for O-edge
mat = AdcMatrix("cvs-adc3", ReferenceState(scfres, core_orbitals=1))
mats = AdcBlockView(mat, "s")
mats.method = mat.method
state = adcc.run_adc(mats, conv_tol=1e-8, n_singlets=10)
state_relaxed = relax_cvs(scfres, "adc3", state)

# Print results
print()
print(state.describe(oscillator_strengths=False))
print()
print()
print(state_relaxed.describe(oscillator_strengths=False))
print()
print("Residual norms: ", state_relaxed.residual_norms)
