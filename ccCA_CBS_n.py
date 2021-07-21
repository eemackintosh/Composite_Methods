from re import DEBUG
import pyscf
import numpy as np
from scipy.optimize import curve_fit
from pyscf.hessian.thermo import thermo, harmonic_analysis
from pyscf.hessian import rks as rks_hess
from pyscf.hessian import uks as uks_hess
from pyscf import dft, lib
from pyscf.geomopt.geometric_solver import optimize
from pyscf.data.elements import N_CORE_SHELLS, ELEMENTS
from pyscf.cc import qcisd_slow as qci
from pyscf.cc import uqcisd_slow as uqci
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

# The ccCA-CBS-n methods are presented and described in detail in
# ref: DeYonker, N. J. J. Chem. Phys. 124, 114104 (2006) DOI: 10.1063/1.2173988


def ccCA_CBS_n(mol_input, temp=298.15):
    """Calculates molecular energy using the composite method ccCA-CBS-1 and ccCA-CBS-2

    Per p. 4 and Table 1 of DeYonker, N. J. J. Chem. Phys. 124, 114104 (2006) DOI: 10.1063/1.2173988,
        E(ccCA-CBS-n) = E_ref + E_QCI + E_CV + E_ZPE
    The first step is geometry optimization using B3LYP/6-31G(d).
    The second step is the calculation of the total number of core orbitals in the molecule.
    Then, the terms of E(ccCA-CBS-n) are calculated in 4 parts:
        PART 1 : E_ref
            Starting energy from Complete Basis Set (CBS) extrapolation
                CBS-1 calls a simple exponential functional
                CBS-2 calls a mixed exponential/Gaussian functional
            MP2(frozen)/CBS aug-cc-PVinfZ
        PART 2 : E_QCI
            Corrects for correlation effects using quadratic configuration interaction
            QCISD(T)/cc-pVTZ - MP2(frozen)/cc-pVTZ
        PART 3 : E_CV
            Corrects for core-valence correlation effects
            MP2(full)/auc-cc-pCVTZ - MP2(frozen)/aug-cc-pVTZ
        PART 4 : E_ZPE
            Corrects for the zero-point vibrational energy
            E_ZPE : B3LYP/6-31G(d)
            Scale Factor : 0.9854
    The terms of E(ccCA-CBS-1) are summed to afford the final energy value.
    
    Note that:
        MP2(frozen) : frozen-core computation
        MP2(full) : includes core-to-valence transitions, PySCF MP2 default
        6-31G(d) : aka 6-31G*

    Parameters
    ----------
    mol_input : object
        Molecule object, including coordinates, spin, charge, and minimal basis

    temp : float, optional
        Temperature in K, by default 298.15

    Returns
    -------
    E_tot : tuple
        (CBS-1, CBS-2) atomic or molecular energy in Hartree
    """
    if mol_input.basis != "sto-3g":
        lib.logger.warn(
            mol_input,
            "The basis set of the molecule object is not sto-3G and will be overwritten in each part of the function.",
        )

    # Geometry Optimization
    # ------------------------------------------------------------------------------------------
    mol_geom = pyscf.M(
        atom=mol_input.atom,
        basis="6-31g*",
        spin=mol_input.spin,
        charge=mol_input.charge,
        output=mol_input.output,
        verbose=mol_input.verbose,
        symmetry=mol_input.symmetry,
    )
    mf = dft.KS(mol_geom).newton()
    mf.xc = "B3LYP"
    mf.kernel()
    mf.stability()
    mf.nuc_grad_method().kernel()

    mol = optimize(
        mf
    )  # returns a mol object with optimized geometry to be used in the following calculations

    # Total Number of Core Orbitals
    # ------------------------------------------------------------------------------------------
    # Required for MP2(frozen) computations

    Z = [
        ELEMENTS.index(xi) for xi in mol.elements
    ]  # list of atomic numbers for each atom in the molecule

    n_core = 0
    for item in Z:
        shell = N_CORE_SHELLS[item]
        s = int(shell[0])  # number of core s orbitals for item
        p = 3 * int(shell[2])  # number of core p orbitals for item
        d = 5 * int(shell[4])  # number of core d orbitals for item
        f = 7 * int(shell[6])  # number of core f orbitals for item
        n_core += s + p + d + f  # total number of core orbitals for the molecule

    lib.logger.info(
        mol_input,
        "The number of core orbitals for frozen core calculations is {}".format(n_core),
    )

    # PART 1 : E_ref
    # ------------------------------------------------------------------------------------------
    zeta = ["d", "t", "q"]  # X in aug-cc-pV{X}Z basis set
    e_results = []
    for x in zeta:
        mol_ref = pyscf.M(
            atom=mol.atom,
            basis="augccpv" + x + "z",
            spin=mol.spin,
            charge=mol.charge,
            output=mol_input.output,
            verbose=mol_input.verbose,
            symmetry=mol_input.symmetry,
        )
        mf = mol_ref.HF().run()
        if mol.elements == ["H"]:
            e_results.append(mf.e_tot)
        else:
            myMP2 = mf.MP2().set(frozen=n_core).run()
            e_results.append(mf.e_tot + myMP2.e_corr)
    e_data = {"aug-cc-pVXZ": np.array([2, 3, 4]), "Energy": np.array(e_results)}

    lib.logger.debug(
        mol_input, "E_ref_MP2_d = {}".format(e_results[0]),
    )
    lib.logger.debug(
        mol_input, "E_ref_MP2_t = {}".format(e_results[1]),
    )
    lib.logger.debug(
        mol_input, "E_ref_MP2_q = {}".format(e_results[2]),
    )

    # CBS-1 Extrapolation
    def CBS_1(x, a, b, c):
        return a + b * np.exp(-c * x)

    popt_CBS_1, _ = curve_fit(
        CBS_1,
        e_data["aug-cc-pVXZ"],
        e_data["Energy"],
        maxfev=2000,
        p0=[e_data["Energy"][2], 1e-3, 1e-3],
        bounds=([-np.inf, 0, 0], [0, np.inf, np.inf]),
    )  # fits CBS_1 function to energy plotted as a function of zeta value
    curve_fit_x = np.linspace(2, 8)
    plt.plot(e_data["aug-cc-pVXZ"], e_data["Energy"], "o-", color="blue", label="data")
    plt.plot(
        curve_fit_x,
        CBS_1(curve_fit_x, *popt_CBS_1),
        "r-",
        color="darkred",
        label=f"CBS-1 fit: a={popt_CBS_1[0]:5.3f}, b={popt_CBS_1[1]:5.3f}, c={popt_CBS_1[2]:5.3f}",
    )
    plt.hlines(popt_CBS_1[0], 2, 8, color="red", label="$E_{\infty}|CBS_1$")

    # CBS-2 Extrapolation
    def CBS_2(x, a, b, c):
        return a + b * np.exp(-(x - 1)) + c * np.exp(-((x - 1) ** 2))

    popt_CBS_2, _ = curve_fit(
        CBS_2,
        e_data["aug-cc-pVXZ"],
        e_data["Energy"],
        maxfev=2000,
        p0=[e_data["Energy"][2], 1e-3, 1e-3],
        bounds=([-np.inf, 0, -np.inf], [0, np.inf, np.inf]),
    )  # fits CBS_2 function to energy plotted as a function of zeta value
    plt.plot(e_data["aug-cc-pVXZ"], e_data["Energy"], "o-", color="blue", label="data")
    plt.plot(
        curve_fit_x,
        CBS_2(curve_fit_x, *popt_CBS_2),
        "r-",
        color="darkgreen",
        label=f"CBS-2 fit: a={popt_CBS_2[0]:5.3f}, b={popt_CBS_2[1]:5.3f}, c={popt_CBS_2[2]:5.3f}",
    )
    plt.hlines(popt_CBS_2[0], 2, 8, color="lime", label="$E_{\infty}|CBS_2$")
    plt.xlabel("Zeta")
    plt.ylabel("E_tot")
    plt.legend()
    plt.savefig("CBS_debug.png")
    plt.close()

    if abs(popt_CBS_1[0] - popt_CBS_2[0]) > 2:
        lib.logger.warn(
            mol_input, "The CBS-1 and CBS-2 extrapolations differ substantially",
        )

    if popt_CBS_1[0] > e_results[2] or popt_CBS_2[0] > e_results[-1]:
        lib.logger.warn(
            mol_input,
            "The extrapolated energy value is greater than that at the maximum zeta value",
        )

    E_ref = (
        popt_CBS_1[0],
        popt_CBS_2[0],
    )  # (CBS-1, CBS-2) extrapolated energy at CBS limit

    lib.logger.info(
        mol_input, "E_ref = {}".format(E_ref),
    )

    # PART 2 : E_QCI
    # ------------------------------------------------------------------------------------------
    mol_QCI = pyscf.M(
        atom=mol.atom,
        basis="ccpvtz",
        spin=mol.spin,
        charge=mol.charge,
        output=mol_input.output,
        verbose=mol_input.verbose,
        symmetry=mol_input.symmetry,
    )
    # QCISD(T)/cc-pVTZ calculation
    mf = mol_QCI.HF().newton()
    mf.conv_tol = 1e-14
    mf.kernel()
    if mol_input.spin != 0:
        mqci = uqci.UQCISD(mf, frozen=n_core)
        mqci.conv_tol = 1e-10
        mqci.kernel()
        e_t = mqci.qcisd_t()
    else:
        mqci = qci.QCISD(mf, frozen=n_core)
        mqci.conv_tol = 1e-10
        mqci.kernel()
        e_t = mqci.qcisd_t()
    E_QCI_calc = e_t + mqci.e_tot

    lib.logger.debug(
        mol_input, "E_QCI_calc = {}".format(E_QCI_calc),
    )

    # MP2/cc-pVTZ calculation
    mf = mol_QCI.HF().run()
    if mol.elements == ["H"]:
        e_results.append(mf.e_tot)
        E_MP2 = mf.e_tot
    else:
        myMP2 = mf.MP2().set(frozen=n_core).run()
        E_MP2 = mf.e_tot + myMP2.e_corr

    lib.logger.debug(
        mol_input, "E_QCI_MP2 = {}".format(E_MP2),
    )

    # E_QCI correction
    E_QCI = E_QCI_calc - E_MP2

    lib.logger.info(
        mol_input, "E_QCI = {}".format(E_QCI),
    )

    # PART 3 : E_CV
    # ------------------------------------------------------------------------------------------
    # full MP2/aug-cc-pCVTZ computation
    if (
        "H" or "He" in mol.elements
    ):  # the aug-cc-pCVTZ does not exist for H or He (no core electrons)
        mol_CV = pyscf.M(
            atom=mol.atom,
            basis={"default": "aug-cc-pCVTZ", "H": "aug-cc-pVTZ", "He": "aug-cc-pVTZ"},
            spin=mol.spin,
            charge=mol.charge,
            output=mol_input.output,
            verbose=mol_input.verbose,
            symmetry=mol_input.symmetry,
        )
    else:
        mol_CV = pyscf.M(
            atom=mol.atom,
            basis="aug-cc-pVTZ",
            spin=mol.spin,
            charge=mol.charge,
            output=mol_input.output,
            verbose=mol_input.verbose,
            symmetry=mol_input.symmetry,
        )
    mf_full = mol_CV.HF().run()
    if mol.elements == ["H"]:
        E_CV_full = mf_full.e_tot
    else:
        myMP2_full = mf_full.MP2().run()
        E_CV_full = mf_full.e_tot + myMP2_full.e_corr

    lib.logger.debug(
        mol_input, "E_CV_full = {}".format(E_CV_full),
    )

    # frozen-core MP2/aug-cc-pCVTZ computation
    mol_CV = pyscf.M(
        atom=mol.atom,
        basis="augccpvtz",
        spin=mol.spin,
        charge=mol.charge,
        output=mol_input.output,
        verbose=mol_input.verbose,
    )
    mf_frozen = mol_CV.HF().run()
    if mol.elements == ["H"]:
        E_CV_frozen = mf_frozen.e_tot
    else:
        myMP2_frozen = mf_frozen.MP2().set(frozen=n_core).run()
        E_CV_frozen = mf_frozen.e_tot + myMP2_frozen.e_corr

    lib.logger.debug(
        mol_input, "E_CV_frozen = {}".format(E_CV_frozen),
    )

    # E_CV correction
    E_CV = E_CV_full - E_CV_frozen

    lib.logger.info(
        mol_input, "E_CV = {}".format(E_CV),
    )

    # PART 4 : E_ZPE
    # ------------------------------------------------------------------------------------------
    mol_ZPE = pyscf.M(
        atom=mol.atom,
        basis="6-31g*",
        spin=mol.spin,
        charge=mol.charge,
        output=mol_input.output,
        verbose=mol_input.verbose,
        symmetry=mol_input.symmetry,
    )
    mf = dft.KS(mol_ZPE).newton()
    mf.xc = "B3LYP"
    mf.kernel()
    mf.stability()
    mf.nuc_grad_method().kernel()
    if mol_input.spin != 0:
        hess = uks_hess.Hessian(mf).kernel()
    else:
        hess = rks_hess.Hessian(mf).kernel()
    freq_au = np.array(
        [
            harmonic_analysis(
                mol_ZPE, hess, exclude_trans=True, exclude_rot=True, imaginary_freq=True
            )["freq_au"]
        ]
    )  # in au, calculated by the thermo.py code
    freq_au *= 0.9854  # scale factor

    thermo_results = thermo(mf, freq_au, temp, pressure=101325)

    E_ZPE = thermo_results["ZPE"][0]

    lib.logger.info(
        mol_input, "E_ZPE = {}".format(E_ZPE),
    )

    # Debug Dataframe
    debug_data = {
        "Method/Basis": [
            "E_ref : MP2(froz)/aug-cc-pVDZ",
            "E_ref : MP2(froz)/aug-cc-pVTZ",
            "E_ref : MP2/aug-cc-pVQZ",
            "E_QCI_calc : QCISD(T)/cc-pVTZ",
            "E_QCI_calc : MP2(froz)/cc-pVTZ",
            "E_CV_full : MP2(full)/aug-cc-pCVTZ",
            "E_CV_frozen : MP2(froz)/aug-cc-pVTZ",
        ],
        "Energy": [
            e_results[0],
            e_results[1],
            e_results[2],
            E_QCI_calc,
            E_MP2,
            E_CV_full,
            E_CV_frozen,
        ],
    }
    df = pd.DataFrame(debug_data)
    df.to_csv("Debug_CBS.csv")

    # Calculation of E(ccCA-CBS-1)
    # ------------------------------------------------------------------------------------------
    E_tot = E_ref + E_QCI + E_CV + E_ZPE  # E_tot = E(ccCA-CBS-n)
    E_tot = tuple(E_tot)
    lib.logger.note(
        mol_input, "E(ccCA-1, ccCA-2) = {}".format(E_tot),
    )
    results = {}
    results["E_CBS1"] = (E_tot[0], "Eh")
    results["E_CBS2"] = (E_tot[1], "Eh")
    results["ZPE"] = (E_ZPE, "Eh")

    # Calculation of Enthalpy Corrections at 298 K and 0 K
    # ------------------------------------------------------------------------------------------
    # Enthalpy components calculated from same thermo.py code used to calculate ZPE in PART 4
    # 298 K
    H_trans = thermo_results["H_trans"][0]
    H_rot = thermo_results["H_rot"][0]
    H_vib = thermo_results["H_vib"][0]
    H_corr_298 = H_trans + H_rot + H_vib
    results["H_corr_298"] = (H_corr_298, "Eh")
    # 0 K
    thermo_results = thermo(mf, freq_au, temperature=0, pressure=101325)
    H_trans = thermo_results["H_trans"][0]
    H_rot = thermo_results["H_rot"][0]
    H_vib = thermo_results["H_vib"][0]
    H_corr_0 = H_trans + H_rot + H_vib
    results["H_corr_0"] = (H_corr_0, "Eh")

    return results


if __name__ == "__main__":
    mol = pyscf.M(
        # atom="""O      0.124849    0.000000   -0.000000
        # H      0.696178    0.000000    0.791387
        # H      0.696178    0.000000   -0.791387""",
        atom="""H 0 0 0""",
        charge=0,
        spin=1,
        basis="sto-3g",
        verbose=5,
        # output="Be.out",
    )

    print(ccCA_CBS_n(mol))

