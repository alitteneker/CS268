import sys, math, random, scipy.misc, itertools, numpy as np, cv2
from matplotlib import pyplot as plt
from operator import attrgetter

# This is from one of the itertools recipes in the online docs
def randomCombination(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted( random.sample( xrange(n), r ) )
    return tuple( pool[i] for i in indices )

def transformPoints(points, t):
    ret = []
    for p in points:
        tp = t * np.asmatrix( p.tolist() ).transpose()
        if tp[2] != 0:
            tp /= tp[2]
        ret.append(tp)
    return ret

def normalizePoints(points, avgDist):
    if len(points) == 0:
        return False, False
    mean = np.array([0,0,0], float)
    for p in points:
        mean += p - np.array([0,0,1])
    mean, dist = mean / len(points), float(0)
    for p in points:
        dist += np.linalg.norm( mean - p )
    t = np.matrix([[ 1, 0, -mean[0] ], [ 0, 1, -mean[1] ], [ 0, 0, len(points) * avgDist / dist ]], float)
    return transformPoints(points, t), t

def eightPointAlgorithm(lfeat, rfeat):
    if len(lfeat) != len(rfeat) and len(lfeat) != 8:
        raise ValueError('Eight point algorithm cannot be run with more than 8 points')
    lf, lt = normalizePoints(lfeat, math.sqrt(2))
    rf, rt = normalizePoints(rfeat, math.sqrt(2))
    A = []
    for i in xrange( 0, len(lfeat) ):
        l, r = lf[i], rf[i]
        A.append( [ l[0]*r[0], l[0]*r[1], l[0], l[1]*r[0], l[1]*r[1], l[1], r[0], r[1], 1 ] )
    u, s, v = np.linalg.svd( np.matrix(A) )
    u, s, v = np.linalg.svd( np.linalg.inv(rt) * v[8].reshape(3,3) * lt )
    return u * np.diagflat(s[0:2].tolist() + [1]) * v

def getFeatures(limg, rimg, numbins):
    sift = cv2.xfeatures2d.SIFT_create()
    lkp, ldes = sift.detectAndCompute(limg, None)
    rkp, rdes = sift.detectAndCompute(rimg, None)
    full_matches = cv2.FlannBasedMatcher( dict( algorithm = 0, trees = 5 ), dict() ).knnMatch(ldes, rdes, k=2)
    hd, wd, bins = float(limg.shape[0]) / numbins, float(limg.shape[1]) / numbins, [ [] for x in xrange( numbins * numbins ) ]
    for i, (m, n) in enumerate(full_matches):
        if m.distance < 0.7 * n.distance:
            p = lkp[ m.queryIdx ].pt
            bins[ int(p[0]/wd) + int(p[1]/hd) * numbins ].append(m)
    lfeat, rfeat = [], []
    for b in bins:
        if len(b) == 0:
            continue
        b.sort( reverse = True, key = attrgetter('distance') )
        lfeat.append( np.array( list(lkp[ b[0].queryIdx ].pt) + [1] ) )
        rfeat.append( np.array( list(rkp[ b[0].trainIdx ].pt) + [1] ) )
    print "Found " + str(len(lfeat)) + " matching features between the images"
    return np.array(lfeat), np.array(rfeat)

def hat(x):
    return np.matrix([ [ 0, -x[2], x[1] ], [ x[2], 0, -x[0] ], [ -x[1], x[0], 0 ] ])

def getFundamentalMatrix(limg, rimg, maxiterations):
    bestF, bestScore, bestMatches = False, 0, []
    lfeat, rfeat = getFeatures(limg, rimg, 10)
    if len(lfeat) < 8:
        raise ValueError("Unable to work with less than 8 matching features")
    feats = xrange(0, len(lfeat))
    for i in xrange(0, maxiterations):
        indices = randomCombination( feats, 8 )
        score, matches = 0, [ [], [] ]
        F = eightPointAlgorithm(lfeat[[indices]], rfeat[[indices]])
        transL = transformPoints(lfeat, F)
        for i in feats:
            if np.linalg.norm( transL[i] - rfeat[i] ) < 5:
                score += 1
                matches[0].append(lfeat[i])
                matches[1].append(rfeat[i])
        #print "Score: " + str(score)
        if score > bestScore:
            bestF = F
            bestScore = score
            bestMatches = matches
    return bestF, bestMatches

def normalize(v):
    l = np.linalg.norm(v)
    if l != 0:
        return v / l
    return v

def calcEpipoles(F):
    u, s, v = np.linalg.svd( F )
    e1 = np.array( v[2, 0:2].tolist() + [1] )
    u, s, v = np.linalg.svd( np.linalg.transpose(F) )
    e2 = np.array( v[2, 0:2].tolist() + [1] )
    return e1, e2

def calcMatchingRectifyingMatrices(limg, rimg, F, matches):
    e1, e2 = calcEpipoles(F)
    center = np.array([ float(rimg.shape[1])/2, float(rimg.shape[0])/2, 1 ])
    rel = e2 - center

    a = -math.acos( np.dot( normalize(rel), np.array([ 1, 0, 0 ]) ) )
    if rel[1] < 0:
        a = -a
    Gt = np.matrix([ [ 1, 0, -center[0] ], [ 0, 1, -center[1] ], [ 0, 0, 1 ] ])
    Gr = np.matrix([ [ math.cos(a), -math.sin(a), 0 ], [ math.sin(a), math.cos(a), 0 ], [ 0, 0, 1 ] ])
    G = np.matrix([ [ 1, 0, 0 ], [ 0, 1, 0 ], [ -1/(Gr*Gt*e2)[0], 0, 1] ])
    H2 = G * Gr * Gt

    R, T_M = np.linalg.qr(F.transpose())
    T = T_M.transpose()[2].getA1()

    factor = np.asmatrix( np.zeros((3*len(matches), 3), np.dtype(float)) )
    result = np.asmatrix( np.zeros((3*len(matches), 1), np.dtype(float)) )
    for i, (x1,x2) in matches:
        currRes = - hat(x2) * hat(T).transpose() * F * np.asmatrix(x1).transpose()
        currFac = hat(x2) * np.asmatrix(T).transpose() * np.asmatrix(x1)
        for j in range(0,3):
            result[ 3*i + j ][0] = currRes[j][0]
            factor[ 3*i + j ] = currFac[j]
    v = np.linalg.inv( factor.transpose() * factor ) * factor.transpose() * result
    H = hat(T).transpose() * F + np.asmatrix(T).transpose() * v.transpose()

    H1 = H2 * H
    return H1, H2

def rectify(left_img_filename, right_img_filename):
    limg, rimg = cv2.imread( left_img_filename, cv2.IMREAD_COLOR ), cv2.imread( right_img_filename, cv2.IMREAD_COLOR )
    print "Loaded images: " + repr(left_img_filename) + " and " + repr(right_img_filename)
    F, matches = getFundamentalMatrix(limg, rimg, 200)
    if F == False:
        print "Unable to find a fundamental matrix between the images"
        return
    ltrans, rtrans = calcMatchingRectifyingMatrices(limg, rimg, F, matches)
    lrect = cv2.warpPerspective( limg, ltrans, (limg.shape[1], limg.shape[0]) )
    rrect = cv2.warpPerspective( rimg, rtrans, (rimg.shape[1], rimg.shape[0]) )
    cv2.imwrite('left_rect.png', lrect)
    cv2.imwrite('right_rect.png', rrect)

rectify("KITTI_data/left_005.png", "KITTI_data/right_005.png")
