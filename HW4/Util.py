import math, random, itertools, numpy as np
from matplotlib import pyplot as plt

# This is from one of the itertools recipes in the online docs
def randomCombination(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted( random.sample( xrange(n), r ) )
    return tuple( pool[i] for i in indices )

def acos(val):
    if val > 1:
        val = 1
    elif val < -1:
        val = -1
    return math.acos(val)

def cross(a, b):
    return np.asmatrix( np.cross( a[0:3].transpose(), b[0:3].transpose() ) ).transpose()

def dot(a, b):
    return np.dot( a.getA1().tolist(), b.getA1().tolist() )

def ndot(a, b):
    l = np.linalg.norm(a) * np.linalg.norm(b)
    if l > 0:
        return dot(a,b) / l
    return 0

def normalize(v):
    l = np.linalg.norm(v)
    if l > 0:
        return v / l
    return np.asmatrix( np.zeros( v.shape, dtype=float ) )

def vector(data):
    return np.matrix( [data], dtype=float ).transpose()

def identity(size):
    return np.asmatrix( np.diag( [ float(1) ] * size ) )

def projectOntoPlane( p, o, n ):
    return p - dot( p - o, n ) * n

def translate(delta):
    ret = identity( delta.shape[0] )
    for i in xrange( 0, delta.shape[0] - 1 ):
        ret[ i, -1 ] = delta[ i, 0 ]
    return ret

def transInterp(Ta, Tb, alpha):
    ra, rb, ta, tb = Ta[:-1,:-1], Tb[:-1,:-1], Ta[:,-1], Tb[:,-1]
    u,s,v = np.linalg.svd( (1-alpha)*ra + alpha*rb )
    r3 = np.asmatrix(u) * np.asmatrix(np.diag([1]*3)) * np.asmatrix(v)
    ret = translate( (1-alpha) * ta + alpha * tb )
    ret[:-1,:-1] = r3
    return ret

def mergeImages(img_bg, img_fg, canvas):
    burn, mask = cv2.threshold( cv2.cvtColor( img_fg, cv2.COLOR_BGR2GRAY ), 0, 255, cv2.THRESH_BINARY )
    bg = cv2.bitwise_and( img_bg, img_bg, mask = cv2.bitwise_not(mask) )
    fg = cv2.bitwise_and( img_fg, img_fg, mask = mask )
    cv2.add(bg, fg, canvas)
    return canvas

def uncollate(data):
    if len(data) == 0:
        return None
    ret = [[]] * len(data[0])
    for s in data:
        for i, el in enumerate(s):
            ret[i].append(el)
    return ret

def RANSAC(dataset, modelSize, buildModel, testModel, distThresh, maxIterations, minVotesSuccess=None, attemptRefinement=True):
    if minVotesSuccess == None:
        minVotesSuccess = len(dataset)
    best_model, best_score, best_data = None, modelSize, []
    for iteration in xrange(0, maxIterations):
        success, score, model, data = attemptModelImprovement(dataset, randomCombination(dataset, modelSize), buildModel, testModel, distThresh, best_score)
        if success:
            print "\t\t\tRANSAC iteration " + str(iteration) + " random combination success ( " + str(score) + " > " + str(best_score) + " )"
            best_model, best_score, best_data = model, score, data
            while True:
                success, score, model, data = attemptModelImprovement(dataset, best_data, buildModel, testModel, distThresh, best_score)
                if success:
                    print "\t\t\tRANSAC immediate refinement success ( " + str(score) + " > " + str(best_score) + " )"
                    best_model, best_score, best_data = model, score, data
                else:
                    print "\t\t\tRANSAC immediate refinement failure ( " + str(score) + " <= "  + str(best_score) + " )"
                    break
            if best_score > minVotesSuccess:
                break
    if best_score > modelSize and attemptRefinement == True:
        success, score, model, data = attemptModelImprovement(dataset, best_data, buildModel, testModel, distThresh, best_score, True)
        if success:
            best_model, best_score, best_data = model, score, data
            print "\t\t\tRANSAC final refinement success: score increased ( " + str(score) + " >= " + str(best_score) + " )"
        else:
            print "\t\t\tRANSAC final refinement failure: score decreased ( " + str(score) + " < "  + str(best_score) + " )"
    return best_model, best_data

def attemptModelImprovement(dataset, modelData, buildModel, testModel, distThresh, bestScore, acceptEqual=False):
    score, model, data = buildAndTest(dataset, modelData, buildModel, testModel, distThresh)
    if score > bestScore or ( acceptEqual and score == bestScore ):
        return True, score, model, data
    return False, score, model, data

def buildAndTest(dataset, modelData, buildModel, testModel, distThresh):
    valid, model = buildModel( modelData )
    if valid == True:
        score, data = testModel( model, dataset, distThresh )
        return score, model, data
    return 0, None, None

def testTransform(trans, matches, THRESH_MAX_DIST):
    votes, match_votes = 0, []
    dists = map( lambda (a,b): np.linalg.norm( a - ( trans * b ) ), matches )
    for i, dist in enumerate(dists):
        if dist <= THRESH_MAX_DIST:
            votes += 1
            match_votes.append( matches[i] )
    return votes, match_votes

def testTriangleSimilarity(features, thresh_dist):
    ( a1, a2 ), ( b1, b2 ), ( c1, c2 ) = features
    ba1, ba2, ca1, ca2, cb1, cb2 = b1-a1, b2-a2, c1-a1, c2-a2, c1-b1, c2-b2
    lba1, lba2, lca1, lca2, lcb1, lcb2 = map( np.linalg.norm, [ ba1, ba2, ca1, ca2, cb1, cb2 ] )
    if abs(lba1-lba2) > thresh_dist or abs(lca1-lca2) > thresh_dist or abs(lcb1-lcb2) > thresh_dist:
        return False
    return True

def buildDirectTransformation(features):
    if len(features) < 3:
        print features
        raise ValueError("Cannot build a direct transformation with fewer than three points (" + str(len(features)) + ")")
    if len(features) == 3 and not testTriangleSimilarity(features, 0.005):
        return False, None
    mean1, mean2 = vector([0]*4), vector([0]*4)
    for ( f1, f2 ) in features:
        mean1 += f1
        mean2 += f2
    mean1, mean2 = mean1 / len(features), mean2 / len(features)
    H = np.zeros( (3,3), dtype=float )
    for ( f1, f2 ) in features:
        H += ( f2 - mean2 )[:3] * ( f1 - mean1 )[:3].transpose()
    u, s, v = np.linalg.svd( H )
    R4 = identity(4)
    R4[ :3, :3 ] = np.asmatrix(v).transpose() * np.asmatrix(u).transpose()
    if np.linalg.det(R4) < 0:
        R4[:,2] *= -1
    trans = translate(mean1) * R4 * translate( -mean2 )
    return True, trans

def normalizePoints(points, normDist = math.sqrt(2) ):
    if len(points) == 0:
        raise ValueError("Cannot normalize a point set of size zero")
    mean = np.zeros(points[0].shape, dtype = float)
    for p in points:
        mean += p
    mean, dist = mean / len(points), float(0)
    for p in points:
        dist += np.linalg.norm( mean - p )
    nd, normed = 1, []
    if dist > 0:
        nd = normDist * len(points) / dist
    t = identity( mean.shape[0] )
    for i in xrange(0, mean.shape[0] - 1):
        t[ i, i  ] = nd
        t[ i, -1 ] = nd * mean[i, 0]
    for p in points:
        normed.append(t * p)
    return t, normed

def buildHomography(features):
    if len(features) < 4:
        raise ValueError('Cannot construct a homography with fewer than four points')
    features1, features2 = uncollate(features)
    t1, feat1 = normalizePoints(features1)
    t2, feat2 = normalizePoints(features2)
    # Generate the matrix for homogeneous least squares
    A = []
    for i in range(featureSize):
        A.append([ -feat2[i][0,0], -feat2[i][1,0], -1,              0,              0,  0, feat1[i][0,0] * feat2[i][0,0], feat1[i][0,0] * feat2[i][1,0], feat1[i][0,0] ])
        A.append([              0,              0,  0, -feat2[i][0,0], -feat2[i][1,0], -1, feat1[i][1,0] * feat2[i][0,0], feat1[i][1,0] * feat2[i][1,0], feat1[i][1,0] ])
    # Solve with an svd, and readjust for normalization
    u, s, v = np.linalg.svd( np.matrix(A) )
    return True, ( np.linalg.inv(t1) * np.asmatrix(v[8].reshape((3,3))) * t2 )
