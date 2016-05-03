import numpy as np, cv2, Util


class Point:
    def __init__(self, pt, color, srcidx = 0):
        if pt.size != 4:
            raise ValueError("Point must be in homogenous R3")
        self.pt = pt
        self.color = color
        self.srcidx = srcidx
    def transform(self, trans):
        self.pt = trans.transform(self.pt)
    def getTransformed(self, trans):
        return Point(trans.transform(self.pt), self.color, self.srcidx)
    def toString(self):
        params = tuple(self.pt[0:3].getA1().tolist()) + tuple(self.color.tolist()[::-1])
        return "%f %f %f %d %d %d" % params


class PointCloud:
    def __init__(self, index = 0):
        self.points = []
        self.index = index
    def addPoint(self, pt):
        self.points.append(pt)
    def addPoints(self, pts):
        self.points += pts
    def addCloud(self, other):
        self.points += other.points
    def transform(self, trans):
        for pt in self.points:
            pt.transform(trans)
    def getTransformed(self, trans):
        ret = PointCloud()
        for p in self.points:
            ret.addPoint(p.transform(trans))
        return ret
    def generateImage(self, K, trans, imgsize):
        img = np.zeros( imgsize + (3,), np.uint8 )
        depths = np.full(imgsize, float('inf'))
        K4 = Util.identity(4)
        K4[:-1,:-1] = K
        fullTrans = K4 * trans
        for p in self.points:
            tp = fullTrans * p.pt
            depth = tp[2, 0]
            if depth <= 0:
                continue
            coord = tuple( ( tp[1::-1] / depth ).getA1().tolist() )
            if coord[0] >= 0 and coord[0] < imgsize[0] and coord[1] >= 0 and coord[1] < imgsize[1] and depth < depths[coord]:
                depths[coord] = depth
                img[coord] = p.color
        return img
    def outputToPLY(self, filename):
        f = open(filename, 'w')
        f.write("ply\nformat ascii 1.0\nelement vertex %d  \n" % len(self.points))
        f.write("property float32 x \nproperty float32 y   \nproperty float32 z  \n")
        f.write("property uchar red \nproperty uchar green \nproperty uchar blue \n")
        f.write("end_header\n")
        for p in self.points:
            f.write(p.toString() + "\n")
        f.close()

def generateSpherePoints(centerPos, radius, albedo, lightPos, numPoints):
    ret = []
    for i in xrange(0,numPoints):
        v = np.random.randn(3)
        n = Util.vector( ( v / np.linalg.norm(v) ).tolist() + [0] )
        p = centerPos + radius * n
        c = max( float(0), Util.ndot( n, lightPos - p ) ) * albedo
        ret.append( Point( p, c ) )
    return ret
