from re import DEBUG
import pyscf
import numpy as np
from pyscf.hessian.thermo import thermo, harmonic_analysis
from pyscf.data.elements import N_CORE_SHELLS, ELEMENTS
from pyscf.data import nist
import pandas as pd
from ccCA_CBS_n import ccCA_CBS_n

df = pd.read_csv("ref_CBS_n.csv")


def dHform_atomization(mol, temp=298.15):
    # PART 1 : Molecular Enthalpy
    # ------------------------------------------------------------------------------------------
    CBS_results = ccCA_CBS_n(mol, temp)
    ZPE = CBS_results["ZPE"][0]
    H_term = CBS_results["H_corr_298"][0] - CBS_results["H_corr_0"][0]
    # CBS-1
    E_CBS1 = CBS_results["E_CBS1"][0]
    H_mol_CBS1 = E_CBS1 + ZPE + H_term
    # CBS-2
    E_CBS2 = CBS_results["E_CBS2"][0]
    H_mol_CBS2 = E_CBS2 + ZPE + H_term
    results = {}
    results["H_mol_CBS1"] = (H_mol_CBS1, "Eh")
    results["H_mol_CBS2"] = (H_mol_CBS2, "Eh")

    # PART 2 : Atomic Enthalpy and Correction
    # ------------------------------------------------------------------------------------------
    H_atom_CBS1_tot = []
    H_atom_CBS2_tot = []
    H_atom_corr = []
    for X in mol.elements:
        # Atomic Enthalpy
        mol = pyscf.M(
            atom="""{} 0 0 0""".format(X),
            charge=df.loc[(df["element"] == "{}".format(X))]["charge"].iloc[0],
            spin=df.loc[(df["element"] == "{}".format(X))]["spin"].iloc[0],
            basis="sto-3g",
            verbose=5,
        )
        CBS_results = ccCA_CBS_n(mol, temp)
        ZPE = CBS_results["ZPE"][0]
        H_term = CBS_results["H_corr_298"][0] - CBS_results["H_corr_0"][0]
        # CBS-1
        E_CBS1 = CBS_results["E_CBS1"][0]
        H_atom_CBS1 = E_CBS1 + ZPE + H_term
        H_atom_CBS1_tot.append(H_atom_CBS1)
        results["H_atom{}_CBS1".format(X)] = (H_atom_CBS1, "Eh")
        # CBS-2
        E_CBS2 = CBS_results["E_CBS2"][0]
        H_atom_CBS2 = E_CBS2 + ZPE + H_term
        H_atom_CBS2_tot.append(H_atom_CBS2)
        results["H_atom{}_CBS2".format(X)] = (H_atom_CBS2, "Eh")
        # Atomic Heat of Formation Correction
        correction = df.loc[(df["element"] == "{}".format(X))]["dHform (eH)"].iloc[0]
        H_atom_corr.append(float(correction))
        results["H_atom{}_correction".format(X)] = (correction, "Eh")

    H_atom_CBS1 = sum(H_atom_CBS1_tot)
    H_atom_CBS2 = sum(H_atom_CBS2_tot)
    H_atom_corr = sum(H_atom_corr)
    H_atom_CBS1 = H_atom_CBS1 - H_atom_corr
    H_atom_CBS2 = H_atom_CBS2 - H_atom_corr

    # PART 3 : dHform
    # ------------------------------------------------------------------------------------------
    dHform_CBS1 = H_mol_CBS1 - H_atom_CBS1
    dHform_CBS2 = H_mol_CBS2 - H_atom_CBS2
    results["dHform_CBS1"] = (dHform_CBS1, "Eh")
    results["dHform_CBS2"] = (dHform_CBS2, "Eh")

    return results


if __name__ == "__main__":
    mol = pyscf.M(
        # atom="""O      0.124849    0.000000   -0.000000
        # H      0.696178    0.000000    0.791387
        # H      0.696178    0.000000   -0.791387""",
        # atom="""Cl 0 0 0""",
        atom="""Li	0.0000	0.0000	0.0000
    Li	0.0000	0.0000	2.6730
    """,
        charge=0,
        spin=0,
        basis="sto-3g",
        verbose=5,
        # output="Be.out",
    )

    print(dHform_atomization(mol))
