import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import FastICA, PCA
from sklearn import decomposition

# Enable plots inside the Jupyter NotebookLet the
%matplotlib inline

image_height=12
noofcomponents=10
image_width=image_height*noofcomponents
image_shape=(image_height,image_width)
noofpixels=image_height*image_width

datafilename="/home/w_khalili/ica/B_challenge_for_Wala'a Khalili.hdf5"



from tables import *

# Create Sample/Row types
class WordImage(IsDescription):
    idnumber  = Int64Col()      # Signed 64-bit integer
    image = Float64Col(image_shape)    # double (double-precision)

# Create ICA dataset for individual student
ImagesFile = open_file(datafilename, mode = "r")
ImagesTable=ImagesFile.root.Images.Images

#print(ImagesTable[1])

 # my code

    
new_data = list()
for x in ImagesTable.iterrows(): # (3000,)
    data = x['image'].reshape(1440,1)
    new_data.append(data)

# ICA Processing 
print("next Pictures are proccesed with ICA ")

stack_new_data = np.hstack(new_data)
ica = FastICA(n_components=10)
S_ica_ = ica.fit_transform(stack_new_data)  # Reconstruct signals
A_ica_ = ica.mixing_  # Get estimated mixing matrix
abs_value_data = np.absolute(S_ica_)

final_data = [abs_value_data[:,y].reshape(12,120) for y in range(10)]
for y in range(10):
    plt.matshow(final_data[y])   
    plt.show()    
    
print("next Pictures are proccesed with PCA ")

#PCA Data Processing 

pca = PCA(n_components=10)
PCA_data = pca.fit_transform(stack_new_data)
abs_value_data = np.absolute(PCA_data)

final_data = [abs_value_data[:,y].reshape(12,120) for y in range(10)]
for y in range(10):
    plt.matshow(final_data[y])   
    plt.show()
    
    

 # Data reading and plotting the result. This is where you should put your own code.


#dataset=np.zeros((ImagesTable.nrows,noofpixels))
#for x in ImagesTable.iterrows(0,10): # Start and stop are specified here, you need to process entire dataset!
#    plt.matshow(x['image'], interpolation='nearest', cmap=cm.jet)
#    plt.show()
        

#ImagesTable.close()


