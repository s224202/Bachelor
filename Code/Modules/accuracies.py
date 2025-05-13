import difflib
def strict_equality(outputs:list[str], targets:list[str]) -> float:
    assert len(outputs) == len(targets), "Target and outputs must be of same length"
    

def partial_match(outputs:list[str], targets:list[str]) -> float:
    assert len(outputs) == len(targets), "Target and outputs must be of same length"
    accs = []
    for i in range(len(outputs)):
        accs.append(difflib.SequenceMatcher(None, outputs[i], targets[i])).ratio()
    return sum(accs)/len(accs)
