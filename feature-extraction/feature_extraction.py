from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np



def feature_extraction():
    file = open("data/names_with_numbers.txt",'r')
    names = []
    for line in file:
        line = line.strip().split()
        if int(line[1]) == 2:
            names.append(line[0])
        if len(names) == 45:
            print("Collected")
            break
        
    #model = VGG16(weights='imagenet', include_top=False)   
    input_shape = (250, 250, 3)
    model = VGG16(weights = 'imagenet', input_shape = (input_shape[0], input_shape[1], input_shape[2]), pooling = 'max', include_top = False)

    features_list = []
    for name in names:
        filename1 = "data/lfw-deepfunneled/lfw-deepfunneled/" + name + "/" + name + "_0001.jpg"
        filename2 = "data/lfw-deepfunneled/lfw-deepfunneled/" + name + "/" + name + "_0002.jpg"
        filenames = [filename1,filename2]
        for image_path in filenames:
            #img = image.load_img(image_path, target_size=(224, 224))
            img = image.load_img(image_path, target_size=(250, 250))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            features = model.predict(x)
            features_list.append(features)
    print("Number of feature vectors:",len(features_list))
    print("Each feature vector is of length:",features_list[0].shape)
    outfile = open("extractions/VGG16_lfw-deepfunneled.txt",'w')
    for features in features_list:
        #print(np.linalg.norm(features))
        features = features/np.linalg.norm(features) #are we allowed to normalize?
        #print(np.linalg.norm(features))
        for i in range(features.shape[1]):
            outfile.write(str(features[0,i]))
            #print(i,str(features[0,i]))
            outfile.write(" ")
        outfile.write("\n")

    outfile.close()

if __name__ == "__main__":
    feature_extraction()
