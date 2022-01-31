import streamlit as st
import os
from PIL import Image



st.title('Chess Position prediction')


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploaded', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1

    except:
        return 0


uploaded_file = st.file_uploader("Upload Image")

if uploaded_file is not None:
    st.image(uploaded_file)
    if st.button("Predict"):
        st.text("Processing...")

        if save_uploaded_file(uploaded_file):
            # display the file
            display_image = Image.open(uploaded_file)
            display_image = display_image.resize((500, 300))
            st.image(display_image)
            prediction = predictor(os.path.join('uploaded', uploaded_file.name))
            prediction = uploaded_file.name
            print(prediction)
            os.remove('uploaded/' + uploaded_file.name)
            # Predictied postion:
            st.image(prediction)





