import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from config import config
from training import inference
#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = config.APP_BG_IMG
    st.image(image_path,use_column_width=True)
    st.markdown(config.ABOUT_APP)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown(config.APP_MARKDOWN)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = inference.predict(test_image)
        #Reading Labels
        st.success(result_index)
