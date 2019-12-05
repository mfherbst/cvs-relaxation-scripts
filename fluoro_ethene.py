#!/usr/bin/env python3
import adcc

from pyscf import gto, scf
from relaxation import relax_cvs

from matplotlib import pyplot as plt

# Run SCF in pyscf
mol = gto.M(
    atom="""
        C    -12.085941752     2.3495203374  -1.38785761e-05
        C     -9.602485779     2.2057983748  2.8408754067e-05
        H    -13.060546886     2.6165393078     1.7515506303
        H    -13.084816123     2.1961242075    -1.7515424237
        F    -8.1639757112     2.3662435971     2.0178565065
        F    -8.1919594188     1.8815343619    -2.0178792432
    """,
    basis={"C": 'cc-pcvdz', "F": 'cc-pvdz', "H": 'cc-pvdz'},
    unit="Bohr",
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-12
scfres.conv_tol_grad = 1e-9
scfres.kernel()

# Run CVS-ADC(2) and relax to ADC(2) for C-edge
state = adcc.cvs_adc2(scfres, conv_tol=1e-8, core_orbitals=4, n_singlets=5)
state_relaxed = relax_cvs(scfres, "adc2", state)

# Print results
print()
print(state.describe())
print()
print()
print(state_relaxed.describe())
print()

state.plot_spectrum(label="CVS-ADC(2)")
state_relaxed.plot_spectrum(label="ADC(2)")
print("Residual norms: ", state_relaxed.residual_norms)

plt.legend()
plt.savefig("fluoro_ethene_relaxed.pdf")
