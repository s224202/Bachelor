import difflib
def strict_equality(outputs:list[str], targets:list[str]) -> float:
    assert len(outputs) == len(targets), "Target and outputs must be of same length"
    sum = 0
    for i in range(len(outputs)):
        s1 = outputs[i].replace(' ', '')
        s2 = targets[i].replace(' ', '')
        s1 = s1.lower()
        s2 = s2.lower()
        if s1 == s2:
            sum += 1
    return sum / len(outputs), [1 if outputs[i] == targets[i] else 0 for i in range(len(outputs))]

def partial_match(outputs:list[str], targets:list[str]) -> float:
    assert len(outputs) == len(targets), "Target and outputs must be of same length"
    accs = []
    for i in range(len(outputs)):
        s1 = outputs[i].replace(' ', '')
        s2 = targets[i].replace(' ', '')
        s1 = s1.lower()
        s2 = s2.lower()
        accs.append(difflib.SequenceMatcher(None,s1, s2).ratio())
    return sum(accs) / len(accs), accs
