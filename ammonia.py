#!/usr/bin/env python3
import adcc

from pyscf import gto, scf
from relaxation import relax_cvs

from matplotlib import pyplot as plt

# Run SCF in pyscf
mol = gto.M(
    atom="""
        N       0.0000000000     0.0000000000     0.1175868000
        H       0.9323800000     0.0000000000    -0.2743692000
        H      -0.4661900000    -0.8074647660    -0.2743692000
        H      -0.4661900000     0.8074647660    -0.2743692000
    """,
    # Use our custom basis for N
    basis={"N": './basis/up6-311++Gss.dat', "H": "6-311++G**"},
    unit="ang"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-12
scfres.conv_tol_grad = 1e-9
scfres.kernel()

# Run CVS-ADC(3) and relax to ADC(3) for N-edge
state = adcc.cvs_adc3(scfres, conv_tol=1e-8, core_orbitals=1, n_singlets=10)
state_relaxed = relax_cvs(scfres, "adc3", state)

# Print results
print()
print(state.describe())
print()
print()
print(state_relaxed.describe())
print()

state.plot_spectrum(label="CVS-ADC(3)")
state_relaxed.plot_spectrum(label="General ADC(3)")
print("Residual norms: ", state_relaxed.residual_norms)

plt.legend()
plt.savefig("ammonia_relaxed.pdf")
