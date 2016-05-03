
  ###########################################
	###   Alan Litteneker - UID 204434197   ###
	###         CS 268 - Homework 2         ###
	###########################################

This folder contains the completed code and an example
resultant image for HW2 (merged.png). The code is in
python, and requires opencv 3 (used for image loading,
feature detection, and correspondence), numpy
(requisite for opencv), and scipy (used for
combinatorics calculations). To run the program,
simply call 'python mosaic.py' from this directory.

The approach taken is as follows:
1) Features (and corresponding descriptors) are found
on each image using SIFT (built in to opencv).
2) Next, the system searches for direct transformations
between images. This begins by finding a set of putative
correspondences between the images using opencv's flann
matcher. These correspondences are then fed into a random
sample consensus to find a homography which maximizes the
number of outliers. This homography is calculated using
normalized direct linear transform.
3) If any direct transformations cannot be identified
between an images, the system searches for a path through the
known direct transformations.
4) All of the direct and combinational transformations are
then searched to find the transformation for each image
to transform it into the coordinate space of the image
at index 0. When more than one transformation was found,
the transformation which has the maximum number of inliers
is used.
5) Finally, all of the images are merged onto a common canvas
(using opencv's warpPerspective for sampling). For disk space
reasons, this merged image is limited in resolution.

Unfortunately, while my program does appear to function properly,
I was not able to get it to finish merging all 7 images by the
deadline. The time required for processing was simply too great.
The file merged.png contains the resultant merging operation of
only 3 of the images.
