from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB5, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import cv2
import pandas as pd


archive_features = np.load('archive/archive_features.npy')
archive_df = pd.read_csv('archive/archive_df.csv')


class FeatureExtractor:
    
    def __init__(self):
        # Use EfficientNetB5 as the architecture and ImageNet for the weight
        base_model = EfficientNetB5(weights='imagenet')
        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

        
    def extract_image(self, img, xmin, ymin, xmax, ymax):
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.zeros((gray_img.shape[0], gray_img.shape[1], 3))
        img[:, :, 0] = gray_img
        img[:, :, 1] = gray_img
        img[:, :, 2] = gray_img
        
        img = img[int(xmin):int(xmax), 
                  int(ymin):int(ymax), : ] 
        
        img = cv2.resize(img, (456, 456))
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        # Extract Features
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)
    
fe = FeatureExtractor()

    
def eucledian_distance(x, y):
    eucl_dist = np.linalg.norm(x - y)
    return eucl_dist


def find_nearest(tfs, trfs, df):
    dists = []
    for i in range(len(trfs)):
        dists.append(eucledian_distance(tfs, trfs[i]))
    df['dist'] = dists
    typs = [df.sort_values('dist').is_princess.iloc[i] for i in range(15)]
    return typs
    

def check_is_princess(image, xmin, ymin, xmax, ymax):    
    features = fe.extract_image(image, xmin, ymin, xmax, ymax)
    is_princess = find_nearest(features, archive_features, archive_df)[0]
    if is_princess == 1:
        return True
    else:
        return False