import streamlit as st
import pandas as pd
import numpy as np
import torch
from PIL import Image
import cv2
from pathlib import Path

from folium.plugins import HeatMap
import folium
from streamlit_folium import folium_static

from google_drive_downloader import GoogleDriveDownloader as gdd

# 'https://drive.google.com/file/d/1rTTqtAyP7AwGNNFMd-cGrx1A0mmO1XEY/view?usp=sharing'


bad_color = (179, 26, 18)
neutral_color = (250, 212, 1)
good_color = (17, 173, 46)

princess_id = 2022

# @st.cache()
# def load_model(path='models/best.pt', device='cpu'):
#     detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/best.pt', force_reload=True)  # default
#     return detection_model

cloud_model_location = '1rTTqtAyP7AwGNNFMd-cGrx1A0mmO1XEY'

@st.cache(ttl=36000, max_entries=1000)
def load_model():
    
    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    f_checkpoint = Path('model/best.pt')
                     
    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            gdd.download_file_from_google_drive(file_id=cloud_model_location,
                                    dest_path=f_checkpoint,
                                    unzip=False)
    
    detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=f_checkpoint, force_reload=True)
    
    return detection_model


def detect_image(image, detection_model):
    
    pred = detection_model(image).pandas().xyxy[0].sort_values('confidence', ascending=False)
    name_ids = []
    for i, n in enumerate(pred.name):
        name_ids.append(n+' '+str(i))
    pred['name_id'] = name_ids
        
    for i in range(len(pred[pred.confidence > 0.5])):
        
        x1 = int(pred.xmin.iloc[i])
        y1 = int(pred.ymin.iloc[i])
        x2 = int(pred.xmax.iloc[i])
        y2 = int(pred.ymax.iloc[i])
        
        if pred.confidence.iloc[i] > 0.9:
            i_color = good_color
        elif pred.confidence.iloc[i] > 0.7:
            i_color = neutral_color
        else:
            i_color = bad_color
        
        image = cv2.rectangle(image, (x1, y1), (x2, y2), i_color, 4)
        image = cv2.putText(image, 
                            '{} - {}%'.format(pred.name_id.iloc[i], round(100*pred.confidence.iloc[i])), 
                            (x1-50, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 3.5, i_color, 5)
        
    return image, pred


def get_animal_idx(pred_data):
    return princess_id
    # return np.random.randint(1000)

    
# <------------------------------------------------------------------------->

# st.set_page_config(layout="wide")

st.title('Защита редких животных')

detection_model = load_model()

st.header('Загрузка данных')
file = st.file_uploader('Загрузите изображение')

if file: # if user uploaded file
    
    image = np.array(Image.open(file))
    
    # process photo  
    processed_image, pred_data = detect_image(image, detection_model)
    
    # print(pred_data)
    
    st.header('Результат детекции')
    
    col1, col2 = st.columns(2)
    col1.metric("Количество тигров", str(pred_data[(pred_data.name == 'Tiger')&(pred_data.confidence > 0.5)].shape[0]))
    col2.metric("Количество леопардов", str(pred_data[(pred_data.name == 'Leopard')&(pred_data.confidence > 0.5)].shape[0]))
    st.image(processed_image)
    
# <------------------------------------------------------------------------->   
    
    st.header('Информация о животных')
    
    options = list(pred_data.name_id.values)
    options.append('Отменить выбор')
    
    option = st.selectbox(
        'Просмотреть информацию по распознанным животным',
        options
    )
    
    if option in list(pred_data.name_id.values):
        
        col3, col4 = st.columns(2)
        
        idx = get_animal_idx(pred_data[pred_data.name_id == option])
        
        col3.metric(label='ID животного', value=idx, 
                    delta='{}%'.format(round(100*pred_data[pred_data.name_id == option].confidence.iloc[0])), 
                                       delta_color="off")
        col4.metric('Имя', 'Принцесса')
        
        col5, col6 = st.columns(2)
        col5.metric('Возраст', 'Взрослый') # взрослый, молодой, пожилой, детеныш
        col6.metric('Болезненность', 'Отсутствует')
    
# <------------------------------------------------------------------------->   
    
    
        df = pd.DataFrame(
            np.random.randn(100, 2) / [2, 2] + [45.37, 136.21],
            columns=['latitude', 'longitude'])
    
        map_heatmap = folium.Map(location=[45.37, 136.21], zoom_start=8)
    
        # Filter the DF for columns, then remove NaNs
        heat_df = df[["latitude", "longitude"]]
        heat_df = heat_df.dropna(axis=0, subset=["latitude", "longitude"])
    
        # List comprehension to make list of lists
        heat_data = [
            [row["latitude"], row["longitude"]] for index, row in heat_df.iterrows()
        ]
    
        # Plot it on the map
        HeatMap(heat_data).add_to(map_heatmap)
    
        # Display the map using the community component
        st.subheader('Тепловая карта перемещений')
        folium_static(map_heatmap)
    
    
    
