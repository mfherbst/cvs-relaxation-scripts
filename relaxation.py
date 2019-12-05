import adcc
import numpy as np

from adcc.guess import guess_zero
from adcc.solver import IndexSpinSymmetrisation
from adcc.solver.preconditioner import JacobiPreconditioner
from adcc.solver.conjugate_gradient import conjugate_gradient, default_print
from adcc.AmplitudeVector import AmplitudeVector


class AdcMatrixShifted(adcc.AdcMatrixlike):
    # Note: This class is not completely inconsistent!

    def __init__(self, matrix, omega=0.0):
        super().__init__(matrix)
        self.update_omega(omega)

    def __matmul__(self, other):
        return super().__matmul__(other) - self.omegamat * other

    def matmul_orig(self, other):
        return super().__matmul__(other)

    def update_omega(self, omega):
        diagonal = AmplitudeVector(*tuple(
            self.diagonal(block) for block in self.blocks
        ))
        self.omega = omega
        self.omegamat = adcc.ones_like(diagonal) * omega


def project_amplitude(cvs, full):
    """
    Project CVS amplitude to a full amplitude
    """
    assert "s" in full.blocks

    n_c, n_v = cvs["s"].shape
    n_c_a = n_c // 2
    # n_v_a = n_v // 2

    n_of_a = full["s"].shape[0] // 2

    # singles
    cvs_s_arr = cvs["s"].to_ndarray()
    cvs_s_arr_a = cvs_s_arr[:n_c_a, :]
    cvs_s_arr_b = cvs_s_arr[n_c_a:, :]

    full_s_arr = np.zeros(full["s"].shape)
    full_s_arr[:n_c_a, :] = cvs_s_arr_a
    full_s_arr[n_of_a:n_of_a + n_c_a, :] = cvs_s_arr_b
    full["s"].set_from_ndarray(full_s_arr, 1e-12)

    if "d" in full.blocks:
        n_o = cvs["d"].shape[0]
        n_o_a = n_o // 2
        # n_of_a = n_c_a + n_o_a

        # doubles
        cvs_d_arr = cvs["d"].to_ndarray()
        cvs_d_arr_aa = cvs_d_arr[:n_o_a, :n_c_a, :, :]
        cvs_d_arr_ab = cvs_d_arr[:n_o_a, n_c_a:, :, :]
        cvs_d_arr_ba = cvs_d_arr[n_o_a:, :n_c_a, :, :]
        cvs_d_arr_bb = cvs_d_arr[n_o_a:, n_c_a:, :, :]

        full_d_arr = np.zeros(full["d"].shape)
        full_d_arr[n_c_a:n_of_a, 0:n_c_a, :, :] =  cvs_d_arr_aa
        full_d_arr[0:n_c_a, n_c_a:n_of_a, :, :] = -cvs_d_arr_aa.transpose((1,0,2,3))
        #
        full_d_arr[n_c_a:n_of_a, n_of_a:n_of_a + n_c_a, :, :] = cvs_d_arr_ab
        full_d_arr[n_of_a:n_of_a + n_c_a, n_c_a:n_of_a, :, :] = -cvs_d_arr_ab.transpose((1,0,2,3))
        #
        full_d_arr[n_of_a + n_c_a:, 0:n_c_a, :, :] = cvs_d_arr_ba
        full_d_arr[0:n_c_a, n_of_a + n_c_a:, :, :] = -cvs_d_arr_ba.transpose((1,0,2,3))
        #
        full_d_arr[n_of_a + n_c_a:, n_of_a:n_of_a + n_c_a, :, :] = cvs_d_arr_bb
        full_d_arr[n_of_a:n_of_a + n_c_a, n_of_a + n_c_a:, :, :] = -cvs_d_arr_bb.transpose((1,0,2,3))

        full["d"].set_from_ndarray(full_d_arr, 1e-12)


def relax_cvs(scfres, method, state, ctol=5e-5):
    singles_block_only = False
    if adcc.AdcMethod(method).level > 1 and "d" not in state.matrix.blocks:
        singles_block_only = True
        # Singles block only method
        refstate = adcc.ReferenceState(scfres)
        origmatrix = adcc.AdcBlockView(adcc.AdcMatrix(method, refstate), "s")
    else:
        refstate = adcc.ReferenceState(scfres)
        origmatrix = adcc.AdcMatrix(method, refstate)
    matrix = AdcMatrixShifted(origmatrix)
    explicit_symmetrisation = IndexSpinSymmetrisation(
        matrix, enforce_spin_kind=state.kind
    )

    assert state.kind == "singlet"
    fullvec = [guess_zero(matrix, spin_block_symmetrisation="symmetric")
               for i in range(len(state.excitation_vectors))]
    for i in range(len(state.excitation_vectors)):
        project_amplitude(state.excitation_vectors[i], fullvec[i])

    preconditioner = JacobiPreconditioner(matrix)

    relaxed_vec = []
    relaxed_ene = []
    residual_norms = []
    for i in range(len(state.excitation_vectors)):
        print("=================")
        print("    State {}".format(i))
        print("=================")
        vec = fullvec[i].copy()  # Not sure this copy is needed, do it for safety
        origvec = vec
        ene = state.excitation_energies[i]
        eneold = ene
        histories = []
        residual = 100000
        xold = vec
        preconditioner.update_shifts(ene - 1e-2)
        matrix.update_omega(ene - 1e-3)
        print("-->  Starting energy {}: {}".format(i, ene))

        for it in range(100):
            eps = 1e-3  # numerical fuzzing to improve conditioning
            if residual > eps / 10 and it > 0:
                matrix.update_omega(ene - eps)
                preconditioner.update_shifts(ene - eps)

            res = conjugate_gradient(
                matrix, rhs=xold, x0=xold, callback=default_print,
                Pinv=preconditioner, conv_tol=ctol / 10,
                max_iter=400,
            )
            x = res.solution
            x = explicit_symmetrisation.symmetrise([x], [origvec])[0]
            x /= np.sqrt(x @ x)

            ene = x @ matrix.matmul_orig(x)
            enediff = ene - eneold
            overlap = np.sqrt(np.abs(origvec @ x))
            resres = matrix.matmul_orig(x) - ene * x
            residual = np.sqrt(resres @ resres)
            print("-->  Energy {}:   {} (enediff: {})"
                  "".format(it, ene, enediff))
            print("-->  Overlap {}:  {}".format(it, overlap))
            print("-->  Residual {}: {}".format(it, residual))

            if np.abs(overlap - 1) > 0.2:
                if not histories:
                    if i == 0:
                        print("     !!! Low overlap detected and got no history"
                              "... trying again")
                        xold = origvec
                    else:
                        raise RuntimeError("Low overlap and got no history.")

                # Pick the energy of the historic result with the best overlap
                # (smallest aberration from 1)
                ene = sorted(histories, key=lambda x: x[1])[0][0] + eps
                print("     !!! Low overlap detected! Changing shift to {:.6g} "
                      "and trying again !!!".format(ene))
                xold = origvec
            elif residual < ctol:
                print("-->   converged")
                break
            else:
                xold = x
                eneold = ene
                histories.append((ene, np.abs(overlap - 1)))

        residual_norms.append(np.sqrt(resres @ resres))
        relaxed_vec.append(x)
        relaxed_ene.append(ene)

    class CvsRelaxationState:
        pass

    res = CvsRelaxationState()
    res.matrix = origmatrix
    res.kind = "singlet"
    res.eigenvectors = relaxed_vec
    res.eigenvalues = np.array(relaxed_ene)

    property_method = None
    if singles_block_only:
        # To not get crashes on property calculation (missing doubles part)
        property_method = "adc1"

    sstate = adcc.ExcitedStates(res, method=method, property_method=property_method)
    sstate.residual_norms = np.array(residual_norms)
    return sstate
