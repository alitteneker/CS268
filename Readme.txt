
    ###########################################
	###   Alan Litteneker - UID 204434197   ###
	###         CS 268 - Homework 1         ###
	###########################################

This folder contains the completed code and an example
resultant image for HW1 (merged.png). The code is in
python, and requires opencv 3 (used for image loading
and display), and numpy (requisite for opencv). To run
the program, simply call 'python mosaic.py' from this
directory.

The approach taken is purely geometric. It first finds
transformations between overlapping images by testing
permutations of translations between points using random
sample consensus. Next, this set of 'edge' transformations
is searched to find a set of paths from each image to a
root image's coordinate system (DFS bounded by a max path
length for efficiency). In order to deal with inaccuracies
in the coordinates of feature points in individual images,
each set of transformations to the image is averaged to
form a transformation consistent across each the entire
merged image.