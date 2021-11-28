from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np
import cv2
import torch
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB5, preprocess_input
from tensorflow.keras.models import Model
from google_drive_downloader import GoogleDriveDownloader as gdd
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="Path to folder with images")
parser.add_argument("-t", "--task", help="Task detect_and_classify/find_princess", 
                    nargs='?', default='detect_and_classify', const='detect_and_classify')
parser.add_argument("-o", "--output", help="Path to output file", 
                    nargs='?', default='labels.csv', const='labels.csv')
args = parser.parse_args()


cloud_model_location = '1rTTqtAyP7AwGNNFMd-cGrx1A0mmO1XEY'
cloud_archive_features = '1zjp8NgTGby5lLCotFZfVMk8cl3TIf--L'
cloud_archive_df = '18Nxx7QiBvCB3UfnXPWo-VhP76Tn1dXLy'


def load_archive():
    save_dest = Path('archive')
    save_dest.mkdir(exist_ok=True)  
    if not Path('archive/archive_features.npy').exists():
        gdd.download_file_from_google_drive(file_id=cloud_archive_features,
                                            dest_path=Path('archive/archive_features.npy'),
                                            unzip=False)
    if not Path('archive/archive_df.csv').exists():
        gdd.download_file_from_google_drive(file_id=cloud_archive_df,
                                            dest_path=Path('archive/archive_df.csv'),
                                            unzip=False)
    return np.load('archive/archive_features.npy'), pd.read_csv('archive/archive_df.csv')

archive_features, archive_df = load_archive()
print('Archive loaded:')

def load_model():
    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    f_checkpoint = Path('model/best.pt')    
    if not f_checkpoint.exists():
        gdd.download_file_from_google_drive(file_id=cloud_model_location,
                                            dest_path=f_checkpoint,
                                            unzip=False)
    detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=f_checkpoint, force_reload=True)
    return detection_model

detection_model = load_model()
print('Model loaded')


def detect_and_classify(image, detection_model):
    pred = detection_model(image).pandas().xyxy[0].sort_values('confidence', ascending=False)
    if pred[pred.confidence > 0.1].shape[0] > 0:
        pred = pred[pred.confidence > 0.1].groupby('name').confidence.count().sort_values(ascending=False).index[0]
        if pred == 'Tiger':
            return 1
        elif pred == 'Leopard':
            return 2
    else:
        return 3

def detect_image(image, detection_model):
    pred = pd.DataFrame(detection_model(image).pandas().xyxy[0].sort_values('confidence', ascending=False))
    pred = pred[pred.confidence > 0.1]
    if pred.shape[0] > 0 and pred['name'].iloc[0] == 'Tiger':
        return True, pred
    else:
        return False, pred
    
    
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
        
        img = img[max(0, int(ymin)):int(ymax), 
                  max(0, int(xmin)):int(xmax), : ] 
        try:
            img = cv2.resize(img, (456, 456))
        except:
            print(img.shape)
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
    

def check_is_princess(image, k=5):
    flag, pred = detect_image(image, detection_model)
    if flag:
        xmin, ymin, xmax, ymax = pred[['xmin', 'ymin', 'xmax', 'ymax']].iloc[0]
        features = fe.extract_image(image, xmin, ymin, xmax, ymax)
        probs = find_nearest(features, archive_features, archive_df)[:k]
        probs_weights = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        prob = 0
        sum_weight = 0
        for i in range(k):
            prob += probs[i] * probs_weights[i]
            sum_weight += probs_weights[i]
        prob /= sum_weight
        if prob > 0.4:
            return 1
        else:
            return 0
    else:
        return 0


path_to_images = args.path
path_to_output = args.output
task = args.task  #  detect_and_classify / find_princess
image_paths = glob(path_to_images+'*.jpg')
result = pd.DataFrame()
result['id'] = [x.split('/')[-1] for x in image_paths]
result_class = []

if task == 'detect_and_classify':
    for path in tqdm(image_paths):
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        result_class.append(detect_and_classify(image, detection_model))
    result['class'] = result_class
    result.to_csv(path_to_output, index=False)
        
elif task == 'find_princess':
    for path in tqdm(image_paths):
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        result_class.append(check_is_princess(image))
    result['class'] = result_class
    result.to_csv(path_to_output, index=False)
        
else:
    print('Wrong task')
