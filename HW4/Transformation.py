import numpy as np

class Transformation:
    def __init__(self, mat, certainty, provenance):
        self.mat = mat
        self.shape = mat.shape
        self.certainty = certainty
        self.provenance = provenance

    def transform(self, point, perspDiv = False):
        t_point = self.mat * point
        if perspDiv:
            t_point /= t_point[-1,0]
        return t_point

    def inverse(self):
        return Transformation( np.linalg.inv(self.mat), self.certainty, self.provenance[::-1] )

    def combine(self, other):
        if self.getDestination() != other.getSource():
            raise ValueError('Mismatch in source and destination of transformation combination')
        return Transformation( other.mat * self.mat, min( self.certainty, other.certainty ), other.provenance[:-1] + self.provenance )

    def getMatrix(self):
        return self.mat

    def getSource(self):
        return self.provenance[-1]

    def getDestination(self):
        return self.provenance[0]

def tryCombination(trans, transforms, dest, max_depth, best_certainty=float('-inf')):
    if trans.getDestination() == dest:
        return [ trans ]
    if len(trans.provenance) > max_depth:
        return []
    found = []
    for step in transforms:
        if step.getDestination() == trans.getDestination() and step.getSource() not in trans.provenance:
            step = step.inverse()
        elif step.getSource() != trans.getDestination() or step.getDestination() in trans.provenance or step.certainty < best_certainty:
            continue
        step_found = tryCombination( trans.combine(step), transforms, dest, max_depth, best_certainty )
        if len(step_found) > 0:
            best_found_certainty = np.max( map( lambda t: t.certainty, step_found ) )
            if best_found_certainty > best_certainty:
                best_certainty = best_found_certainty
            found += step_found
    return found

def searchForPaths(dest, sources, transforms, max_depth):
    combinations = []
    for source in sources:
        for trans in transforms:
            if trans.getDestination() == source and trans.getSource() != dest:
                trans = trans.inverse()
            elif trans.getSource() != source or trans.getDestination() == dest:
                continue
            found = tryCombination(trans, transforms, dest, max_depth)
            if len(found) > 0:
                combinations += found
    return combinations

def getBestTransformations(dest, sources, transforms):
    ret = []
    for source in sources:
        bestTrans = None
        for trans in transforms:
            if trans.getDestination() == dest and trans.getSource() == source and ( bestTrans == None or trans.certainty > bestTrans.certainty ):
                bestTrans = trans
        if bestTrans == None:
            raise ValueError("Unable to find a transformation from " + str(source) + " to " + str(dest))
        ret.append(bestTrans)
    return ret
