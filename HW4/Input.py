import csv, numpy as np, Image


def readInput(filename):
    f = open(filename, 'r')
    print "Loading data from file " + filename
    images, mode, K = [], "NONE", np.asmatrix(np.diag([1]*3))
    while True:
        line = f.readline()
        if not line or line[:-1] == "END":
            break
        if line.isspace():
            continue
        line = line[:-1]
        if line.find("K_MATRIX") == 0:
            K = np.asmatrix( map( float, line.split(' ')[1:] ) ).reshape((3, 3))
        elif line == "BEGIN IMAGE_FILENAMES":
            mode = "IMAGE_FILENAMES"
        elif line == "END IMAGE_FILENAMES":
            mode = "NONE"
        elif mode == "IMAGE_FILENAMES":
            images.append( Image.Image( *line.split(' ') ) )
        else:
            print "Unrecognized input line: " + repr(line)
    return K, images

def readCSVtoMatrix(filename, delimiter=","):
    data = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
        for row in reader:
            data.append( map(float, row) )
    return np.matrix(data)
