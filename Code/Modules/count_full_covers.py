import pandas as pd
from translator import translate
iupac = pd.read_csv('~/Sem6/Bachelor/Data/glycan_sequences_iupac_extended.csv')
smiles = pd.read_csv('~/Sem6/Bachelor/Data/glycan_sequences_smiles_isomeric.csv')
wurcs = pd.read_csv('~/Sem6/Bachelor/Data/glycan_sequences_wurcs.csv')

# Count the glytoucan ids that are present in all three datasets
glytoucan_ids_iupac = set(iupac['glytoucan_ac'])
glytoucan_ids_smiles = set(smiles['glytoucan_ac'])
glytoucan_ids_wurcs = set(wurcs['glytoucan_ac'])

full_names = []
for i in glytoucan_ids_iupac:
    if i in glytoucan_ids_smiles and i in glytoucan_ids_wurcs:
        full_names.append(i)

## build new csv file will all system names and glytoucan ids that are present in all three datasets
full_names_df = pd.DataFrame(full_names, columns=['glytoucan_ac'])
full_names_df['iupac'] = [iupac[iupac['glytoucan_ac'] == i]['sequence_iupac_extended'].values[0] for i in full_names]
full_names_df['smiles'] = [smiles[smiles['glytoucan_ac'] == i]['sequence_smiles_isomeric'].values[0] for i in full_names]
full_names_df['wurcs'] = [wurcs[wurcs['glytoucan_ac'] == i]['sequence_wurcs'].values[0] for i in full_names]
full_names_df['selfies'] = [translate(smiles[smiles['glytoucan_ac'] == i]['sequence_smiles_isomeric'].values[0]) for i in full_names]
full_names_df.to_csv('~/Sem6/Bachelor/Data/glycan_sequences_full_covers.csv', index=False)

