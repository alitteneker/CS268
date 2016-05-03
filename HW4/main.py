print "Loading modules . . ."

import sys, math, scipy.misc, numpy as np, cv2
import Util, Input, Image, Point, Transformation
from matplotlib import pyplot as plt

def reconstruction(K, images):
    inv_K = np.linalg.inv(K)

    # find all direct transformations
    directTransforms = []
    max_img_src = 1
    for i, (imgA) in enumerate(images):
        for imgB in images[ i + 1 : i + 1 + max_img_src ]:

            print "\tLooking for direct transformation from " + str(imgB.index) + " to " + str(imgA.index)

            matPoints = imgA.matchesToDeepPoints(imgB, imgA.buildMatches(imgB, True), inv_K)
            print "\t\tMatches found: " + str(len(matPoints))

            trans, vote_matches = Util.RANSAC( matPoints, 3, Util.buildDirectTransformation, Util.testTransform, 0.002, 10000, 0.5 * len(matPoints) )

            if len(vote_matches) > 0:
                directTransforms.append( Transformation.Transformation( trans, len(vote_matches), [ imgA.index, imgB.index ] ) )
                print "\t\tDirect transformation found!"
            else:
                print "\t\tUnable to find a direct transformation!"

    print "\tDirect transformation search completed. Searching for combinational paths . . ."
    combinationTransforms = Transformation.searchForPaths(0, xrange( 1, len(images) ), directTransforms, 27)

    cloudTransforms = [ Transformation.Transformation(Util.identity(4), 1, [0,0]) ] + Transformation.getBestTransformations( 0, xrange(1, len(images)), directTransforms + combinationTransforms )

    print "\tAll transformations identified. Combining point clouds . . ."
    pointCloud = Point.PointCloud()
    for i, img in enumerate(images):
        print "\t\tAdding points from image " + str(i)
        points = img.getAllPoints( inv_K, cloudTransforms[i].getMatrix() )
        pointCloud.addPoints(points)
    print "\tPoint cloud combination finished with " + str(len(pointCloud.points)) + " total points"

    return pointCloud, cloudTransforms

def manipulation(pointCloud):
    startSize = len(pointCloud.points)

    # TODO: it would be a lot better to parameterize this in the input file . . . but laziness . . .
    print "\tManipulation: Adding two spheres"
    pointCloud.addPoints( Point.generateSpherePoints( centerPos=Util.vector([0,-0.1,0.9,1]), radius=0.05, lightPos=Util.vector([0,-2,0,1]), albedo=np.array([255,0,0]), numPoints=20000 ) )
    pointCloud.addPoints( Point.generateSpherePoints( centerPos=Util.vector([0.2,0.1,1.15,1]), radius=0.1, lightPos=Util.vector([0,-2,0,1]), albedo=np.array([0,255,0]), numPoints=70000 ) )

    print "\tManipulation added " + str( len(pointCloud.points) - startSize ) + " points"

def hallucination(K, pointCloud, transforms):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Annoying. Why is the size of the video writer frame opposite from everywhere else in cv2?
    out = cv2.VideoWriter('output.mov', fourcc, 20.0, (640, 480), True)

    frameCount, interpFrames = 0, 5
    print "\tStarting video generation of " + str( ( interpFrames - 1 ) * ( len(transforms) - 1 ) + 1 ) + " frames"
    for i in xrange(0, len(transforms)-1):
        Ta, Tb = transforms[i], transforms[i+1]
        for f in xrange(0, interpFrames-1):
            frameCount += 1
            print "\t\tGenerating frame " + str(frameCount)
            frame = pointCloud.generateImage( K, Util.transInterp( Ta, Tb, float(f) / interpFrames ), (480, 640) )
            out.write(frame)
    frameCount += 1
    print "\t\tGenerating frame " + str(frameCount)
    frame = pointCloud.generateImage( K, transforms[-1], (480, 640) )
    out.write(frame)

    out.release()
    print "\tGenerated video saved to output.mov"

def main(inputFile):

    # Parse the input file, loading images, depth data, etc.
    K, images = Input.readInput(inputFile)
    print "Loaded " + str(len(images)) + " images from file " + inputFile

    # Reconstruction: Combine each image's point cloud into a common point cloud
    pointCloud, transforms = reconstruction(K, images)
    print "Reconstruction finished!"

    # Manipulation: Add to or modify the contents of the point cloud
    manipulation(pointCloud)
    print "Manipulation finished!"

    # Hallucination: Generate new data from the composite point cloud
    hallucination( K, pointCloud, map( lambda t: np.linalg.inv( t.mat ), transforms ) )
    print "Hallucination finished!"

    print "Final step: Saving point cloud to disk . . ."
    pointCloud.outputToPLY("points.ply")

main("input.txt")
print "Huzzah: Program completed with no errors!"
