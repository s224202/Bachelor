import numpy as np
import scipy.stats as stats
import itertools
file = open("./Data/partial_match_scores.txt", "r")
lines = file.readlines()
file.close()
nomen_to_index = {'selfies': 0, 'smiles': 1, 'wurcs': 2, 'iupac': 3}
when_used_as_origin = [[], [], [], []] ## This is basically a numpy array
when_used_as_target = [[], [], [], []]
for i, line in enumerate(lines):
    nomens, accs = line.split(": ")
    orig_nomen, target_nomen = nomens.split(" to ")
    accs = accs.strip("]\n")
    accs = accs.strip("[")
    accs = accs.split(", ")
    accs = [float(acc) for acc in accs]
    accs = np.array(accs)
    #print(f"Is {orig_nomen} to {target_nomen} normal? {stats.shapiro(accs)[1]}")
    when_used_as_origin[nomen_to_index[orig_nomen]].append(accs)
    when_used_as_target[nomen_to_index[target_nomen]].append(accs)

# ## Summary stats for when used as origin

# print("When used as origin:")
# for i, nomen in enumerate(nomen_to_index.keys()):
#     accs = when_used_as_origin[i]
#     accs = np.array(accs)
#     print(f"{nomen}:")
#     print(f"Mean: {np.mean(accs):.2f}")
#     print(f"Median: {np.median(accs):.2f}")
#     print(f"Std: {np.std(accs):.2f}")
#     print(f"Min: {np.min(accs):.2f}")
#     print(f"Max: {np.max(accs):.2f}")
#     print(f"p-value of non-normality: {stats.shapiro(accs)[1]:.2f}")
#     print()
## Summary stats for when used as target
# print("When used as target:")
# for i, nomen in enumerate(nomen_to_index.keys()):
#     accs = when_used_as_target[i]
#     accs = np.array(accs)
#     print(f"{nomen}:")
#     print(f"Mean: {np.mean(accs):.2f}")
#     print(f"Median: {np.median(accs):.2f}")
#     print(f"Std: {np.std(accs):.2f}")
#     print(f"Min: {np.min(accs):.2f}")
#     print(f"Max: {np.max(accs):.2f}")
#     print(f"p-value of non-normality: {stats.shapiro(accs)[1]}")
#     print()

## Since the distribution is not normal, we will use the Wilcoxon signed-rank test to compare the means of the two groups
## Also none of these variable names make sense. 
print("Wilcoxon signed-rank test results:")
pvals = []
for (orig, orig_2) in itertools.permutations(nomen_to_index.keys(), 2):
    orig_idx = nomen_to_index[orig]
    orig_2_idx = nomen_to_index[orig_2]
    accs_orig = when_used_as_target[orig_idx]
    accs_orig_2 = when_used_as_target[orig_2_idx]
    accs_orig = np.array(accs_orig)
    accs_orig_2 = np.array(accs_orig_2)
    accs_orig =accs_orig.flatten()
    accs_orig_2 = accs_orig_2.flatten()
    diff = accs_orig - accs_orig_2
    pval = stats.wilcoxon(diff, alternative='greater').pvalue
    pvals.append(pval)
    print(f"{orig} is a better target than {orig_2}: {pval:.2f}")

print(stats.false_discovery_control(pvals, method='bh'))