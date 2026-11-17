"""
Reynaldo Bautista
Intro to Machine Learning
"""

import numpy as np
import sklearn.preprocessing as pprocessing
import math

#Created a numpy array to later apply some special operations on it
myMatrix = np.array([[14.34, 15.7, 19.98, 20.83 ],
                    [21.82, 24.04, 23.24, 17.89],
                    [24.8, 23.95, 22.61, 27.54 ],
                    [20.13, 20.68, 24.86725275, 25.34907692]])
print("Original matrix: \n", myMatrix)

#Binarized the matrix using the mean of the matrix as the threshold for consistency
binarizedMatrix = pprocessing.Binarizer(threshold=(np.mean(myMatrix))).transform(myMatrix)
print("Binarized matrix \n",binarizedMatrix)

#Calculated the variance of the data. Here I opted to use sample variance since this is a small
#set of data
variance = np.var(myMatrix, ddof=1)
print("Variance: ",variance)

#Normalized the data usign l1 and l2 normalization, same as we did in class
normalizedMatrixL1 = pprocessing.normalize(myMatrix, norm='l1')
normalizedMatrixL2 = pprocessing.normalize(myMatrix, norm='l2')
print("Normalized data L1: \n", normalizedMatrixL1, "\n Nomralized data L2 \n", normalizedMatrixL2)

#Created labels for the sky and created an encoder object
LabelsForSky = ["Dark sky", "Star Studded Sky"]
encoder = pprocessing.LabelEncoder()

#Gave the labels to the encoder and now "Dark sky" will be 0 and "Star Studded Sky" will be 1
encoder.fit(LabelsForSky)

#I reshaped the binarized matrix as a 1D array and casted all its values as integers
#The reason is that I noticed it wouldn't let me encode it size my values, though binarized,
#where still float type, and the array was 2D when encoder only accepts 1D arrays.
flatBinarizedMatrix = binarizedMatrix.reshape(-1).astype(int)
#Then I transformed the numeric values into letters and print them
encodedBinarizedMatrix = encoder.inverse_transform(flatBinarizedMatrix)
print("Encoded Matrix: \n",encodedBinarizedMatrix.reshape(4,4))

#Tranformed the letters back to numbers, and I had to cast the normal array into a "np.array" since
#"encoder.transform" returns a normal array. 
#It isn't necesary but helps visualize it better in terminal
decodedBinarizedMatrix = encoder.transform(encodedBinarizedMatrix)
decodedBinarizedMatrix = np.array(decodedBinarizedMatrix)
print("Decoded Matrix: \n",decodedBinarizedMatrix.reshape(4,4))