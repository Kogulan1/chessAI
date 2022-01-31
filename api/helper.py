import os
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array


current_path = os.getcwd()
model_path = os.path.join(current_path, 'model.h5')
predictor_model = load_model("model.h5")
with open(model_path, 'rb') as handle:
    dog_breeds = pickle.load('model.h5')

# feature_extractor = load_model(r'static\feature_extractor.h5')
from keras.applications.resnet_v2 import ResNet50V2 , preprocess_input as resnet_preprocess
from keras.applications.densenet import DenseNet121, preprocess_input as densenet_preprocess
from keras.layers.merge import concatenate
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input
from keras.models import Model

input_shape = (25,25,3)
input_layer = Input(shape=input_shape)


#first extractor inception_resnet
preprocessor_resnet = Lambda(resnet_preprocess)(input_layer)
inception_resnet = ResNet50V2(weights = 'imagenet',
                                     include_top = False,input_shape = input_shape,pooling ='avg')(preprocessor_resnet)

preprocessor_densenet = Lambda(densenet_preprocess)(input_layer)
densenet = DenseNet121(weights = 'imagenet',
                                     include_top = False,input_shape = input_shape,pooling ='avg')(preprocessor_densenet)


merge = concatenate([inception_resnet,densenet])
feature_extractor = Model(inputs = input_layer, outputs = merge)

print('model loaded')
def predictor(img_path): # here image is file name

    img = load_img(img_path, target_size=(25,25))
    # print(path)
    # img = cv2.resize(img,(331,331))
    img = img_to_array(img)
    img = np.expand_dims(img,axis = 0)
    features = feature_extractor.predict(img)
    prediction = predictor_model.predict(features)*100
    prediction = pd.DataFrame(np.round(prediction,1),columns = dog_breeds).transpose()
    prediction.columns = ['values']
    prediction  = prediction.nlargest(5, 'values')
    prediction = prediction.reset_index()
    prediction.columns = ['name', 'values']
    return(prediction)