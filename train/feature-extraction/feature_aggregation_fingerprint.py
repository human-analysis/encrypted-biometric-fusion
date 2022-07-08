
import numpy as np
import torch

features_list = []
labels = []
for i in range(1,2001):
    for letter in ["f","s"]:
        if i < 10:
            index = "000" + str(i)
        elif i < 100:
            index = "00" + str(i)
        elif i < 1000:
            index = "0" + str(i)
        else:
            index = str(i)
        filename = "extractions/NISTSD4_512x512_dp_normalized/" + letter + index + ".npy"
        point = np.load(filename)
        point = torch.tensor(point)
        print(torch.linalg.norm(point))
        features_list.append(point)
        labels.append(index)
print("Number of feature vectors:",len(features_list))
print("Each feature vector is of length:",features_list[0].shape)
#outfile = open("extractions/VGGFace_vgg_lfw-deepfunneled.txt",'w')
outfile = open("extractions/fingerprints.txt",'w')
for features in features_list:
    #features = features/np.linalg.norm(features) #are we allowed to normalize?
    for i in range(features.shape[0]):
        print(str(float(features[i])))
        outfile.write(str(float(features[i])))
        outfile.write(" ")
    outfile.write("\n")

outfile2 = open("extractions/fingerprints_labels.txt",'w')
for label in labels:
    outfile2.write(str(int(label)))
    outfile2.write("\n")

outfile.close()
outfile2.close()
