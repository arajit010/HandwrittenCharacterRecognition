import streamlit as st
import tensorflow as tf
from tensorflow import keras
import cv2

st.set_page_config(page_title="Cat_Dog-Classification", page_icon=":shark:", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.set_option('deprecation.showfileUploaderEncoding', False)
hide_st_style="""
    <style>
        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}
    </style>
"""
st.markdown(hide_st_style,unsafe_allow_html=True)
@st.cache_resource
def load_model():
    model=keras.models.load_model('model_hand.h5')
    return model

model=load_model()

st.title("Cat-Dog Image Recognition")

file=st.file_uploader("Please upload a image",type=["jpg","png"])
from PIL import Image, ImageFilter 
import numpy as np
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

def import_and_predict(img, model):
    
    # Convert to RGB
    #img = Image.fromarray(img)
    img = img.convert('RGB')

    # Resize to (400, 400)
    img = img.resize((400, 400))

    # Apply Gaussian Blur
    img_copy = img.filter(ImageFilter.GaussianBlur(radius=7))

    # Convert to grayscale
    img_gray = img_copy.convert('L')

    # Apply thresholding
    img_thresh = img_gray.point(lambda x: 0 if x < 100 else 255, '1')

    # Resize to (28, 28)
    img_final = img_thresh.resize((28, 28))

    # Convert to NumPy array
    img_final = np.array(img_final)

    # Reshape for prediction
    img_final = img_final.reshape((1, 28, 28, 1))

    # Make prediction using the model
    img_pred = word_dict[np.argmax(model.predict(img_final))]

    return img_pred

if file is None:
    st.markdown("##### There is no image")
else:
    img=Image.open(file)
    st.image(img,use_column_width=True)
    predictions=import_and_predict(img, model)
    num=np.argmax(predictions)
    message=(f'Predicted Value is : {word_dict[num]}')
    val=message.title()
    st.success(val)