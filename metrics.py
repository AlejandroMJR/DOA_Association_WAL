from itertools import combinations

def compute_rca(groundTruth, estimatedAssignments):
    """
    Computes the rca metric as defined in the paper

    Args:
    groundTruth: list of tuples of ground truth assignments
    estimatedAssignments: list of tuples of estimated assignments

    Returns:
    rca: percentage of correct associations
    """
    numCorrect = sum(1 for est in estimatedAssignments if est in groundTruth)
    totalAssignments = len(groundTruth)
    rca = numCorrect / totalAssignments * 100
    return rca
def compute_rcpa(groundTruth, estimatedAssignments):
    """
    Computes the rcpa metric as defined in the paper

    Args:
    groundTruth: list of tuples of ground truth assignments
    estimatedAssignments: list of tuples of estimated assignments

    Returns:
    rcpa: percentage of correct pairwise associations
    """
    totalPairs = 0
    correctPairs = 0

    numNodes = len(estimatedAssignments[0])  # Number of nodes (columns in assignment tuples)

    for estTuple in estimatedAssignments:
        for (i, j) in combinations(range(numNodes), 2):
            if estTuple[i] is not None and estTuple[j] is not None:  # Consider only valid assignments
                totalPairs += 1
                for gtTuple in groundTruth:
                    if estTuple[i] == gtTuple[i] and estTuple[j] == gtTuple[j]:
                        correctPairs += 1
                        break
    rcpa = (correctPairs / totalPairs) * 100 if totalPairs > 0 else 100
    return rcpa