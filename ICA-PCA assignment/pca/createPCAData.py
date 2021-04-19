import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from fontdemo import Font
import csv

sample_height=12
sample_width=12
maxchars=10
sample_shape=(sample_height,sample_width*maxchars)
pca_discount_factor=0.95
sigma_noise=10
noofimagesamples=3000
codeLength=10
studentIDLength=6

from tables import *

# Create Sample/Row types
class WordImage(IsDescription):
    idnumber  = Int64Col()      # Signed 64-bit integer
    image = Float64Col(sample_shape)    # double (double-precision)
    
class MetaData(IsDescription):
    studentid  = StringCol(studentIDLength)
    frozenimage = Float64Col(sample_shape)    # double (double-precision)
    imageorder = Int64Col((codeLength))    # double (double-precision)   
    

# Create Database, that is open a file in "w"rite mode
MetaDataFile = open_file('PCA_challenge_MetaData.hdf5', mode = "w")
MetaGroup=MetaDataFile.create_group("/", 'meta', 'Meta Information for Lecturer')
MetaTable=MetaDataFile.create_table(MetaGroup, 'studentimage', MetaData, 'Meta Information for Lecturer')

# Get character images
fnt=Font('DejaVuSerif.ttf', 12)

with open('Encode1516.txt', 'rb') as csvfile:
    
    
    mds_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    mds_reader.next()               # Skip headers
    for row in mds_reader:
        mu_noise=np.random.randn(*sample_shape) # Frozen noise
        order=np.random.permutation(codeLength) 
        chars=list()
        normalizations=list()
        for character in row[2]:
            ch=fnt.render_character(character)
            chars.append(ch)
            normalizations.append(np.sqrt(12*120)/np.sqrt(np.sum(ch.pixels))) # Without normalization larger characters dominate the variance
        
        # Write meta data to table
        metarow=MetaTable.row
        metarow['studentid']  = row[1]
        metarow['frozenimage'] = mu_noise    # double (double-precision)
        metarow['imageorder'] = order    # double (double-precision)
        
        metarow.append()
        MetaTable.flush()
        
        # Create PCA dataset for individual student
        ImagesFile = open_file('PCA_challenge_for_'+ row[0]+'.hdf5', mode = "w")
        
        ImagesGroup=ImagesFile.create_group("/", 'Images', 'Images for PCA assignment')
        ImagesTable=ImagesFile.create_table(ImagesGroup, 'Images', WordImage, 'Images for PCA assignment')

        for imageNo in np.arange(noofimagesamples):
            x= sigma_noise * np.random.randn(*sample_shape) + mu_noise
            for charNo in np.arange(len(chars)):
                ch = chars[charNo]
                N=normalizations[charNo]
                loc= order[charNo]
                x[:ch.height,loc*sample_width:loc*sample_width+ch.width]+=N*np.random.randn(1)*np.array(np.reshape(ch.pixels,(ch.height,ch.width)))*np.power(pca_discount_factor,charNo)
            
            # Write meta data to table
            imagerow=ImagesTable.row
            imagerow['idnumber'] = imageNo
            imagerow['image'] = x
            imagerow.append()
        
            #plt.matshow(x, interpolation='nearest', cmap=cm.bone)
            #plt.show()  
            
        ImagesTable.flush()
        ImagesTable.close()
 
