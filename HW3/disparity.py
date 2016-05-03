import sys, math, random, scipy.misc, itertools, numpy as np, cv2
from matplotlib import pyplot as plt

def getPatch(img, row, col, size):
    return img[ row - size : row + size + 1, col - size : col + size + 1 ].astype(float)

def distanceHeuristic(score, allScores, patchSize):
    pdelta = math.sqrt( 3 * ( 1 + patchSize * 2 )**2 )
    meanScore = np.mean(allScores)
    stdScore = np.std(allScores)
    if score < 1.5 * pdelta:
        return True
    if score < 10 * pdelta and score < meanScore - 0.5 * stdScore:
        return True
    # if score < 15 * pdelta and score < meanScore - stdScore:
    #     return True
    return False

def scanForMatch(lpatch, rimg, lineIndex, patchSize, startRight, maxSearch):
    bestIndex, bestDist, distances = -1, float("inf"), []
    for index in xrange( startRight, min( rimg.shape[1] - patchSize, startRight + maxSearch ) ):
        rpatch = getPatch(rimg, lineIndex, index, patchSize)
        dist = np.linalg.norm( lpatch - rpatch )
        distances.append(dist)
        if dist < bestDist: # and min( bestDist - dist, dist ) > 20:
            bestIndex = index
            bestDist = dist
    # print "\t\t" + str(startRight) + ", best " + str(bestIndex) + ": " + str(bestDist) + " < " + str(np.mean(distances)) + " (" + str(np.std(distances)) + ")"
    if bestIndex > 0 and distanceHeuristic(bestDist, distances, patchSize):
        return bestIndex
    return False

def scanImages(limg, rimg, patchSize, maxSearch):
    disparity = np.zeros( ( limg.shape[0], limg.shape[1] ), np.uint8 )
    for line in xrange(patchSize, limg.shape[0]):
        startRight = patchSize
        goodCount, badCount, isFirst, disp = 0, 0, True, []
        for leftIndex in xrange( patchSize, limg.shape[1] - patchSize):
            if isFirst:
                maxS = 500
                isFirst = False
            else:
                maxS = maxSearch
            rightIndex = scanForMatch( getPatch(limg, line, leftIndex, patchSize), rimg, line, patchSize, startRight, maxS)
            if rightIndex == False:
                badCount += 1
            else:
                goodCount += 1
                startRight = rightIndex
                difference = round( abs( leftIndex - rightIndex ) )
                disp.append(difference)
                disparity[line, leftIndex] = difference
        print "\tLine " + str(line) + " avg disparity " + str(np.mean(disp)) + ", matches " + str(goodCount) + " - " + str(badCount) + " (" + str(float(goodCount)/(goodCount+badCount)) + ")"
    return disparity

def main(leftFilename, rightFilename, ps, maxSearch):
    limg = cv2.copyMakeBorder( cv2.imread(leftFilename, cv2.IMREAD_COLOR), ps, ps, ps, ps, cv2.BORDER_REPLICATE )
    rimg = cv2.copyMakeBorder( cv2.imread(rightFilename, cv2.IMREAD_COLOR), ps, ps, ps, ps, cv2.BORDER_REPLICATE )

    print "Loaded left " + repr(leftFilename) + " and right " + repr(rightFilename)

    disparity = cv2.equalizeHist( scanImages(limg, rimg, ps, maxSearch)[ ps : limg.shape[0] - ps, ps : limg.shape[1] - ps ] )

    cv2.imwrite("disparity.png", disparity)

    plt.imshow(disparity, cmap = 'gray')
    plt.show()

main("images/im6_small.ppm", "images/im2_small.ppm", 8, 50)
