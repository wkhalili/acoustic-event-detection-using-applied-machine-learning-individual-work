import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

image_height=12
noofcomponents=10
image_width=image_height*noofcomponents
image_shape=(image_height,image_width)
noofpixels=image_height*image_width

datafilename='ICA_challenge_for_Jan Jager.hdf5'

from tables import *

# Create Sample/Row types
class WordImage(IsDescription):
    idnumber  = Int64Col()      # Signed 64-bit integer
    image = Float64Col(image_shape)    # double (double-precision)

# Create ICA dataset for individual student
ImagesFile = open_file(datafilename, mode = "r")
ImagesTable=ImagesFile.root.Images.Images


'''
    Data reading and plotting the result. This is where you should put your own code.

'''
dataset=np.zeros((ImagesTable.nrows,noofpixels))
for x in ImagesTable.iterrows(0,8): # Start and stop are specified here, you need to process entire dataset!
    plt.matshow(x['image'], interpolation='nearest', cmap=cm.jet)
    plt.show()
    

              

ImagesTable.close()

