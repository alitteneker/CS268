
  ###########################################
	###   Alan Litteneker - UID 204434197   ###
	###         CS 268 - Homework 3         ###
	###########################################

This folder contains the completed code for HW 3. The
code is in python, and requires opencv 3 (used for
image io, feature detection, and matching), and numpy
(requisite for opencv). To run the program, simply
call 'python mosaic.py' from this directory.

The approach taken is as follows:
1) Camera calibration: calibrate.pyplo
This is straightforward. It follows the opencv camera
calibration tutorial code pretty exactly.
2) Image rectification: rectification.py
My code for this does NOT appear to work (ie. major
unknown bug), and I am not completely sure why.
The approach, however, would be as follows:
Calculate matching features, find a fundamental matrix
(using a normalized eight point algorithm), calculate
matching homographies, and warp.
3) Dense correspondence and disparity: disparity.py
This is a rather naive formulation. It works by running
a naive patch scan from scan lines on the right image
to corresponding lines on the left image, and outputs
a disparity image. As such, it falls prey to many local
minima.

Unfortunately, while the calibration and disparity portions of
my code appear to be working, the rectification has some issues
I have not been able to work out. As a result, this work is
currently incomplete, much to my disappointment.

I have included a sample of the calibration and disparity
image outputs.
