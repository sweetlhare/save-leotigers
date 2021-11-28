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


bad_color = (179, 26, 18)
neutral_color = (250, 212, 1)
good_color = (17, 173, 46)

cloud_model_location = '1rTTqtAyP7AwGNNFMd-cGrx1A0mmO1XEY'
cloud_archive_features = '1zjp8NgTGby5lLCotFZfVMk8cl3TIf--L'
cloud_archive_df = '18Nxx7QiBvCB3UfnXPWo-VhP76Tn1dXLy'
cloud_princess_tracking = '12NA-F-Qhm1zKB1lTT3mqejhcUtv0LclM'

@st.cache(ttl=36000, max_entries=1000)
def load_archive():
    
    save_dest = Path('archive')
    save_dest.mkdir(exist_ok=True)
                     
    with st.spinner("Downloading archive data... this may take awhile! \n Don't stop it!"):
        if not Path('archive_features.npy').exists():
            gdd.download_file_from_google_drive(file_id=cloud_archive_features,
                                                dest_path=Path('archive/archive_features.npy'),
                                                unzip=False)
    
    with st.spinner("Downloading archive data... this may take awhile! \n Don't stop it!"):
        if not Path('archive_df.csv').exists():
            gdd.download_file_from_google_drive(file_id=cloud_archive_df,
                                                dest_path=Path('archive/archive_df.csv'),
                                                unzip=False)
            
    with st.spinner("Downloading archive data... this may take awhile! \n Don't stop it!"):
        if not Path('archive_df.csv').exists():
            gdd.download_file_from_google_drive(file_id=cloud_princess_tracking,
                                                dest_path=Path('archive/princess_tracking.csv'),
                                                unzip=False)
            
    return True

print(load_archive())
from princess_identification import check_is_princess


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


def get_animal_idx(image, pred_data):
    xmin, ymin, xmax, ymax = pred_data[['xmin', 'ymin', 'xmax', 'ymax']].iloc[0]
    return check_is_princess(image, xmin, ymin, xmax, ymax)

    
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
        
        flag, prob = get_animal_idx(image, pred_data[pred_data.name_id == option])
        
        st.metric('Вероятность того, что выбранное животное - тигрица Принцесса', '{}%'.format(prob))
        
        if flag:
            
            df = pd.read_csv('archive/princess_tracking.csv', sep=';')
            
            st.metric('Имя', df.name.iloc[0])
            
            col5, col6 = st.columns(2)
            col5.metric('Возраст', '{} лет'.format(df.age.iloc[0])) # взрослый, молодой, пожилой, детеныш
            col6.metric('Болезненность', 'Отсутствует')
    
# <------------------------------------------------------------------------->   
        
            map_heatmap = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=9)
        
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
            heat_freq = st.checkbox('Тепловая карта частоты перемещения')
            if heat_freq:
                folium_static(map_heatmap)
            
        else:
            st.markdown(f'<p style="color:#B34746;font-size:40px;">Выбранное животное - не тигрица Принцесса</p>', unsafe_allow_html=True)
    
    
    
