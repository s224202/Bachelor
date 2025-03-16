import selfies
enc = selfies.encoder

def translate(smiles):
    return enc(smiles)

print(translate('C[C@@H]1O[C@@H](O[C@H]2[C@H](O[C@H]3[C@H](O)[C@@H](NC(C)=O)[C@H](OC[C@@H](O)[C@H](O)[C@H](O[C@@H]4O[C@H](CO)[C@H](O)[C@H](O)[C@H]4O)[C@H](CO)NC(C)=O)O[C@@H]3CO)O[C@H](CO)[C@H](O)[C@@H]2O[C@H]2O[C@H](CO)[C@H](O)[C@H](O)[C@H]2NC(C)=O)[C@@H](O)[C@H](O)[C@@H]1O'))