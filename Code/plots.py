import matplotlib.pyplot as plt

file = open("./Data/partial_match_scores.txt", "r")
lines = file.readlines()
file.close()
fig, axs = plt.subplots(4, 4)
nomen_to_index = {'selfies': 0, 'smiles': 1, 'wurcs': 2, 'iupac': 3}
for i, line in enumerate(lines):
    nomens, accs = line.split(": ")
    orig_nomen, target_nomen = nomens.split(" to ")
    ax = axs[nomen_to_index[orig_nomen], nomen_to_index[target_nomen]]
    accs = accs.strip("]\n")
    accs = accs.strip("[")
    accs = accs.split(", ")
    accs = [float(acc) for acc in accs]
    ax.hist(accs, bins=20)
    ax.set_title(f"{orig_nomen} to {target_nomen}")
    ax.set_xlim(0, 1)
for i in range(4):
    axs[i,i].set_visible(False)
    axs[i,0].set_ylabel("Frequency")
    axs[3,i].set_xlabel("partial match score")
plt.suptitle("Partial Match Scores")

plt.tight_layout()
plt.show()