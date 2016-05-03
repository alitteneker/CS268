import numpy as np, cv2, Input, Point, Util

FLANN_INDEX_KDTREE = 0
class Image:
    INDEX_GEN = 0

    # Call anything within some distance in pixels a match
    THRESH_MAX_DIST = 4

    sift = cv2.xfeatures2d.SIFT_create()
    flann = cv2.FlannBasedMatcher( dict( algorithm = FLANN_INDEX_KDTREE, trees = 5 ), dict() )

    def __init__(self, img_fn, depth_fn=False):
        self.imgData = cv2.imread(img_fn, cv2.IMREAD_COLOR)
        if depth_fn:
            self.depthData = Input.readCSVtoMatrix(depth_fn)
            if self.imgData.shape[1] != self.depthData.shape[1] or self.imgData.shape[0] != self.depthData.shape[0]:
                raise ValueError("Image and depth data do not have the same dimensions")
        else:
            self.depthData = False
        self.width = self.imgData.shape[1]
        self.height = self.imgData.shape[0]
        self._buildFeatures()
        self.index = Image.INDEX_GEN
        Image.INDEX_GEN += 1
        print "\tLoaded image " + repr(img_fn) + " to index " + str(self.index) + " and found " + str(len(self.kp)) + " features"

    def _buildFeatures(self):
        self.kp, self.des = Image.sift.detectAndCompute(self.imgData, None)
        # TODO: should we be binning here?

    def buildMatches(self, other, filterDepth=False):
        full_matches = Image.flann.knnMatch(self.des, other.des, k=2)
        # Filter the initial matches down to a more manageable putative set using Lowe's simple ratio test
        matches = []
        for i,(m,n) in enumerate(full_matches):
            if m.distance < 0.7 * n.distance:
                # if specified filter matches so that we only include matches which have depth data for both images
                if not filterDepth or ( self.depthData[ tuple( self.kp[ m.queryIdx ].pt )[::-1] ] != 0 and other.depthData[ tuple( other.kp[ m.trainIdx ].pt )[::-1] ] != 0 ):
                    matches.append(m)
        return matches

    def getPointCloud(self, inv_K, trans):
        ret = Point.PointCloud()
        ret.addPoints( self.getAllPoints(inv_K, trans) )
        return ret

    def getAllPoints( self, inv_K, trans=Util.identity(4) ):
        ret = []
        for x in xrange(0, self.width):
            for y in xrange(0, self.height):
                depth = self.depthData[y,x]
                if depth != 0:
                    wp = self.pixCoordToDeep( [x, y], inv_K)
                    ret.append( Point.Point( trans * wp, self.imgData[y,x], self.index ) )
        return ret

    def pixCoordToDeep(self, coord, inv_K):
        cp = inv_K * np.matrix([ coord + [1] ], float).transpose()
        return np.append( self.depthData[ coord[1], coord[0] ] * cp, [[1]], 0 )

    def matchesToImgPoints(self, other, matches):
        ret = []
        for match in matches:
            myFeat    = np.matrix( list( self.kp[  match.queryIdx ].pt ) + [1], dtype=float ).transpose()
            otherFeat = np.matrix( list( other.kp[ match.trainIdx ].pt ) + [1], dtype=float ).transpose()
            ret.append( ( myFeat, otherFeat ) )
        return ret

    def matchesToDeepPoints(self, other, matches, inv_K):
        ret = []
        for match in matches:
            myFeat    = self.pixCoordToDeep(  list( self.kp[  match.queryIdx ].pt ), inv_K )
            otherFeat = other.pixCoordToDeep( list( other.kp[ match.trainIdx ].pt ), inv_K )
            ret.append( ( myFeat, otherFeat ) )
        return ret
