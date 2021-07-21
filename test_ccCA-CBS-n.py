import pyscf
import numpy as np
import pandas as pd
import ccCA_CBS_n
from pyscf.data.elements import ELEMENTS
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

df = pd.read_csv("ref_CBS_n.csv")

# mol = pyscf.M(atom="Li 0 0 0", charge=0, spin=1, basis="sto-3g", verbose=3,)
# print(mol.spin)


# spin = []
# no_spin = []
# for item in df["item"]:
#     s = df["spin"]
#     if s[item] == 0:
#         no_spin.append(df["element"][item])
#         print(df["element"][item])
#     else:
#         spin.append(df["element"][item])
# print("spin = ", len(spin))
# print("no spin = ", len(no_spin))

# CBS_1

myCBS_1 = []
ref_CBS_1 = []
elements = []
for item in df["item"]:
    s = df["spin"]
    c = df["charge"]
    # if s[item] == 0 and c[item] == 0:
    #     mol_test = pyscf.M(
    #         atom="{} 0 0 0".format(df["element"][item]),
    #         charge=df["charge"][item],
    #         spin=df["spin"][item],
    #         basis="sto-3g",
    #         verbose=3,
    #     )
    #     myCBS_1.append(ccCA_CBS_n.ccCA_CBS_n(mol_test, "CBS_1"))
    #     ref_CBS_1.append(df["CBS-1"][item])
    #     elements.append(df["element"][item])
    if item != 0 and c[item] == 0:
        mol_test = pyscf.M(
            atom="{} 0 0 0".format(df["element"][item]),
            charge=df["charge"][item],
            spin=df["spin"][item],
            basis="sto-3g",
            verbose=3,
        )
        myCBS_1.append(ccCA_CBS_n.ccCA_CBS_n(mol_test)[0])
        ref_CBS_1.append(df["CBS-1"][item])
        elements.append(df["element"][item])
myCBS_1 = np.array(myCBS_1)

myCBS_2 = []
ref_CBS_2 = []
for item in df["item"]:
    s = df["spin"]
    c = df["charge"]
    # if s[item] == 0 and c[item] == 0:
    #     mol_test = pyscf.M(
    #         atom="{} 0 0 0".format(df["element"][item]),
    #         charge=df["charge"][item],
    #         spin=df["spin"][item],
    #         basis="sto-3g",
    #         verbose=3,
    #     )
    #     myCBS_2.append(ccCA_CBS_n.ccCA_CBS_n(mol_test, "CBS_2"))
    #     ref_CBS_2.append(df["CBS-2"][item])
    if item != 0 and c[item] == 0:
        mol_test = pyscf.M(
            atom="{} 0 0 0".format(df["element"][item]),
            charge=df["charge"][item],
            spin=df["spin"][item],
            basis="sto-3g",
            verbose=3,
        )
        myCBS_2.append(ccCA_CBS_n.ccCA_CBS_n(mol_test)[1])
        ref_CBS_2.append(df["CBS-2"][item])

myCBS_2 = np.array(myCBS_2)

Z = [
    ELEMENTS.index(xi) for xi in elements
]  # list of atomic numbers for each atom in the molecule

comparison = {
    "system": elements,
    "atomic number": Z,
    "my_CBS_1": myCBS_1,
    "ref_CBS_1": ref_CBS_1,
    "diff_CBS_1": abs(ref_CBS_1 - myCBS_1),
    "my_CBS_2": myCBS_2,
    "ref_CBS_2": ref_CBS_2,
    "diff_CBS_2": abs(ref_CBS_2 - myCBS_2),
}
df = pd.DataFrame(comparison)
df.to_csv("CBS-n_comparison.csv")

# CBS-1 Plot
x = df["atomic number"]
y = df["diff_CBS_1"]
plt.scatter(x, y, color="red")
plt.xlabel("Atomic Number")
plt.ylabel("CBS-1 Error")
plt.savefig("CBS_1_Error.png")
plt.close()

# CBS-2 Plot
x = df["atomic number"]
y = df["diff_CBS_2"]
plt.scatter(x, y, color="lime")
plt.xlabel("Atomic Number")
plt.ylabel("CBS-2 Error")
plt.savefig("CBS_2_Error.png")


exit(0)


def test_CBS_1():
    for item in df["item"]:
        s = df["spin"]
        if s[item] == 0:
            mol_test = pyscf.M(
                atom="{} 0 0 0".format(df["element"][item]),
                charge=df["charge"][item],
                spin=df["spin"][item],
                basis="sto-3g",
                verbose=3,
            )
            np.testing.assert_almost_equal(
                ccCA_CBS_n.ccCA_CBS_n(mol_test, "CBS_1"),
                df["CBS-1"][item],
                1,
                "{} test failed".format(
                    str(df["element"][item]) + str(df["charge"][item])
                ),
            )
            # np.testing.assert_almost_equal(
            #     ccCA_CBS_n.ccCA_CBS_n(mol_test, "CBS_1"),
            #     -14.6302325,
            #     3,
            #     "{} test failed".format(
            #         str(df["element"][item]) + str(df["charge"][item])
            #     ),
            # )
        else:
            print("Spinny boi")


if __name__ == "__main__":
    test_CBS_1()
