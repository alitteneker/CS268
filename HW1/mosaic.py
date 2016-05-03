import sys, math, random, scipy.misc, itertools, numpy as np, cv2
from matplotlib import pyplot as plt

# This is from one of the itertools recipes in the online docs
def randomCombination(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted( random.sample( xrange(n), r ) )
    return tuple( pool[i] for i in indices )

class Transformation:
    MAX_DEPTH = 3
    def __init__(self, mat, provenance):
        self.mat = mat
        self.provenance = provenance
    def transform(self, point):
        if isinstance(point, np.ndarray):
            point = point.tolist()
        t_point = ( self.mat * np.matrix( point + [1] ).transpose() ).getA1()
        return Transformation.perspectiveDivision(t_point)
    def inverse(self):
        return Transformation( np.linalg.inv(self.mat), self.provenance[::-1] )
    def combine(self, other):
        if self.getDestinationIndex() != other.getSourceIndex():
            raise ValueError('Mismatch in source and destination of transformation combination')
        return Transformation(other.mat * self.mat, other.provenance[:-1] + self.provenance)
    def getSourceIndex(self):
        return self.provenance[-1]
    def getDestinationIndex(self):
        return self.provenance[0]
    def addDelta(self, delta):
        self.mat = np.matrix([[ 1, 0, delta[0] ], [ 0, 1, delta[1] ], [ 0, 0, 1 ]]) * self.mat

    @classmethod
    def perspectiveDivision(cls, vec):
        if vec[2] != 0:
            return ( vec / vec[2] )[0:2]
        return vec[0:2]

    @classmethod
    def buildHomography(cls, features1, features2):
        featureSize = len(features1)
        if len(features2) != featureSize:
            raise ValueError('Unequal number of features provided for homography construction')
        if featureSize < 4:
            raise ValueError('Fewer than four features provided for homography construction')
        # Calculate the centroid of the feature points for normalization
        mean1, mean2 = (np.array([0,0]), np.array([0,0]))
        for i in range(featureSize):
            mean1 += features1[i]
            mean2 += features2[i]
        mean1 /= featureSize
        mean2 /= featureSize
        # Calculate the average distance from the centroid for normalization
        dist1, dist2 = (0, 0)
        for i in range(featureSize):
            dist1 += np.linalg.norm( mean1 - features1[i] )
            dist2 += np.linalg.norm( mean2 - features2[i] )
        dist1 /= math.sqrt(2) * featureSize
        dist2 /= math.sqrt(2) * featureSize
        # Normalize the feature points of each image to a new coordinate space, with an average distance of sqrt(2) from the centroid at the origin
        t1 = np.matrix([[ 1, 0, -mean1[0] ], [ 0, 1, -mean1[1] ], [ 0, 0, 1/dist1 ]])
        t2 = np.matrix([[ 1, 0, -mean2[0] ], [ 0, 1, -mean2[1] ], [ 0, 0, 1/dist2 ]])
        feat1, feat2 = ([], [])
        for i in range(featureSize):
            feat1.append( Transformation.perspectiveDivision( ( t1 * np.matrix( features1[i].tolist() + [1] ).transpose() ).getA1() ) )
            feat2.append( Transformation.perspectiveDivision( ( t2 * np.matrix( features2[i].tolist() + [1] ).transpose() ).getA1() ) )
        # Generate the matrix for homogeneous least squares
        A = []
        for i in range(featureSize):
            A.append([ -feat2[i][0], -feat2[i][1], -1,            0,            0,  0, feat1[i][0] * feat2[i][0], feat1[i][0] * feat2[i][1], feat1[i][0] ])
            A.append([            0,            0,  0, -feat2[i][0], -feat2[i][1], -1, feat1[i][1] * feat2[i][0], feat1[i][1] * feat2[i][1], feat1[i][1] ])
        # Solve for H
        u, s, v = np.linalg.svd( np.matrix(A) )
        untransformed_H = np.matrix([ v[8, 0:3].getA1(), v[8, 3:6].getA1(), v[8, 6:9].getA1() ])
        return ( np.linalg.inv(t1) * untransformed_H * t2 )

    @classmethod
    def _tryCombination(cls, trans, transforms, destImg):
        if trans.getDestinationIndex() == destImg.index:
            return [ trans ]
        if len(trans.provenance) > Transformation.MAX_DEPTH:
            return []
        found = []
        for step in transforms:
            if step.getDestinationIndex() == trans.getDestinationIndex() and step.getSourceIndex() not in trans.provenance:
                step = step.inverse()
            elif step.getSourceIndex() != trans.getDestinationIndex() or step.getDestinationIndex() in trans.provenance:
                continue
            found += cls._tryCombination(trans.combine(step), transforms, destImg)
        return found

    @classmethod
    def searchForPaths(cls, startImg, destImg, transforms):
        combinations = []
        for trans in transforms:
            if trans.getDestinationIndex() == startImg.index and trans.getSourceIndex != destImg.index:
                trans = trans.inverse()
            elif trans.getSourceIndex() != startImg.index or trans.getDestinationIndex() == destImg.index:
                continue
            found = cls._tryCombination(trans, transforms, destImg)
            if len(found) > 0:
                combinations += found
        return combinations

FLANN_INDEX_KDTREE = 0
class Image:
    # Call anything within some distance in pixels a match
    THRESH_MAX_DIST = 4
    # If we happen across a transform which can get a sizable percentaget of our putative matches to correspond, immediately call it a success
    THRESH_VOTE_PERCENT = 0.4
    INDEX_GEN = 0
    sift = cv2.xfeatures2d.SIFT_create()
    flann = cv2.FlannBasedMatcher( dict( algorithm = FLANN_INDEX_KDTREE, trees = 5 ), dict() )
    def __init__(self, filename):
        self.imgData = cv2.imread(filename, cv2.IMREAD_COLOR)
        self.width = self.imgData.shape[1]
        self.height = self.imgData.shape[0]
        self._buildFeatures()
        self.index = Image.INDEX_GEN
        Image.INDEX_GEN += 1
        print "\tLoaded image " + repr(filename) + " to index " + str(self.index) + " and found " + str(len(self.kp)) + " features"
    def _buildFeatures(self):
        self.kp, self.des = Image.sift.detectAndCompute(self.imgData, None)
    def _buildMatches(self, other):
        full_matches = Image.flann.knnMatch(self.des, other.des, k=2)
        # Filter the initial matches down to a more manageable putative set using Lowe's simple ratio test
        matches = []
        for i,(m,n) in enumerate(full_matches):
            if m.distance < 0.7 * n.distance:
                matches.append(m)
        return matches
    def contains(self, point):
        return point[0] >= 0 and point[0] < self.width and point[1] >= 0 and point[1] < self.height
    def _getTransform(self, other, matches):
        myFeats, otherFeats = self._matchesToPoints(other, matches)
        return Transformation( Transformation.buildHomography(myFeats, otherFeats), [self.index, other.index] )
    def _testTransform(self, trans, other, matches):
        votes, match_votes = 0, []
        for match in matches:
            myFeat = np.array( self.kp[match.queryIdx].pt )
            otherFeat = trans.transform( list(other.kp[match.trainIdx].pt) )
            if np.linalg.norm( myFeat - otherFeat ) <= Image.THRESH_MAX_DIST:
                votes += 1
                match_votes.append(match)
        return votes, match_votes
    def _matchesToPoints(self, other, matches):
        reta, retb = [], []
        for match in matches:
            reta.append( np.array( self.kp[  match.queryIdx ].pt ) )
            retb.append( np.array( other.kp[ match.trainIdx ].pt ) )
        return reta, retb
    def findTransform(self, other):
        matches = self._buildMatches(other)
        print "\t\tFound " + str(len(matches)) + " putative matches"
        if len(matches) < 10:
            return False
        max_combs = scipy.misc.comb(len(matches), 4) / 8
        thresh_votes = len(matches) * Image.THRESH_VOTE_PERCENT
        i, best_score, best_trans, best_matches = 0, 10, False, []
        while i < max_combs:
            i += 1
            comb = randomCombination(matches, 4)
            trans = self._getTransform(other, comb)
            score, match_votes = self._testTransform(trans, other, matches)
            if score > best_score:
                best_score = score
                best_trans = trans
                best_matches = match_votes
                print "\t\t\tAttempt " + str(i) + " : votes " + str(score)
            if best_score >= thresh_votes:
                break
        if best_trans:
            print "\t\tSuccess with " + str(len(best_matches)) + " votes (" + str( 100 * float(len(best_matches)) / len(matches) ) + "%)"
            return self._getTransform(other, best_matches)
        return false

def findTransformations(images):
    transforms = []
    # Start out by looking for direct correspondence transformations between images
    for i in range( 0, len(images) ):
        for j in range( i+1, len(images) ):
            print "\tSearching for direct transformation from " + str(j) + " to " + str(i)
            trans = images[i].findTransform(images[j])
            if trans:
                transforms.append(trans)
            else:
                print "\t\tNo reasonable direct transformation found"
    imageTransforms = [ Transformation( np.matrix([[1,0,0],[0,1,0],[0,0,1]], np.dtype(float)), [0,0] ) ]
    for i in range(1, len(images)):
        counter = 0
        for trans in transforms:
            if trans.getSourceIndex() == i and trans.getDestinationIndex() == 0:
                if counter == 0:
                    imageTransforms.append(trans)
                counter += 1
        if counter == 0:
            # TODO: find a combinational path
            raise ValueError('Unable to find any transformation from image ' + str(i) + ' to image 0')
    return imageTransforms

def mergeImages(images, transforms):
    # Calculate the size of the final merged image
    minP, maxP = ( np.array([0, 0]), np.array([0, 0]) )
    for trans in transforms:
        width, height = images[trans.getSourceIndex()].width, images[trans.getSourceIndex()].height
        for point in ([0, 0], [0, height], [width, 0], [width, height]):
            current = trans.transform(point)
            for j in (0, 1):
                if current[j] < minP[j]:
                    minP[j] = current[j]
                if current[j] > maxP[j]:
                    maxP[j] = current[j]
    if minP[0] < 0 or minP[1] < 0:
        delta = -minP.clip(None, 0)
        for trans in transforms:
            trans.addDelta( delta )
    size = [ maxP[0] - minP[0], maxP[1] - minP[1] ]
    canvas = np.zeros( ( size[1], size[0], 3), np.uint8 )
    for i in range( 0, len(images) ):
        print "\tMerging image " + str(i)
        warped = cv2.warpPerspective( images[i].imgData, transforms[i].mat, ( size[0], size[1] ) )
        ret, mask = cv2.threshold( cv2.cvtColor( warped, cv2.COLOR_BGR2GRAY ), 0, 255, cv2.THRESH_BINARY )
        canvas_bg = cv2.bitwise_and( canvas, canvas, mask = cv2.bitwise_not(mask) )
        warped_fg = cv2.bitwise_and( warped, warped, mask = mask )
        cv2.add(canvas_bg, warped_fg, canvas)
    return canvas

def readInput(filename):
    images = []
    f = open(filename, 'r')
    print "Loading images from file " + filename
    while True:
        line = f.readline()
        if not line or line[0:-1] == "END":
            break
        if line.isspace():
            continue
        img = Image( line[0:-1] )
        images.append(img)
    return images

def main(inputFile, outputFile):

    # Parse the input file, which should contain the name of an image on each line
    images = readInput(inputFile)
    print "Loaded " + str(len(images)) + " images from file " + inputFile

    # Now we do the real work:
    #   1) identify a transformation for each image on to a common coordinate system
    #   2) use the transformations to merge each image in to a composite mosaic image
    imageTransforms = findTransformations(images)
    print "Transformations identified for all images"

    mergedImage = mergeImages(images, imageTransforms)
    print "Mosaic image generated. Shape (height, width, channels): " + str(mergedImage.shape)

    # Write the merged composit
    cv2.imwrite(outputFile, mergedImage)
    print "Mosaic image saved to " + outputFile

    # Finally, show the image
    plt.imshow(mergedImage)
    plt.show()

main("input.txt", "merged.png")
