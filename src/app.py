import cv2
import numpy as np
import streamlit as st
from auth import authenticate
from image_handlers import verify_image_liveness, search_image_deformities, check_similarity
from aws_client import client

st.markdown("<h1 style='text-align: center; '>Login</h1>", unsafe_allow_html=True)

user = None
authenticated = False
uploaded_file = None

with st.form("my_form"):
    name = st.text_input("Username")
    password = st.text_input("Password", type="password")

    st.markdown("<p>Suspicious activity has been reported on this account recently, to confirm its authenticity take a "
                "photo so we can perform facial recognition.</p>", unsafe_allow_html=True)

    # noinspection PyRedeclaration
    uploaded_file = st.camera_input(label="Your webcam", label_visibility="hidden")

    submitted = st.form_submit_button("Submit")
    if submitted:
        if not name or not password or not uploaded_file:
            st.error("Please enter username, password and photo")
        else:
            user = authenticate(name, password)
            if len(user) == 0:
                st.error("Invalid credentials")
            else:
                user = user[0]
                authenticated = True

if uploaded_file is not None and authenticated and user is not None:
    file_buffer = bytearray(uploaded_file.getvalue())

    try:
        loading = st.empty()
        loading.write("Processing face recognition...")

        response = client.detect_faces(
            Image={'Bytes': file_buffer},
            Attributes=["ALL"]
        )

        search_image_deformities(response)
        check_similarity(user, file_buffer)
        label, acc = verify_image_liveness(uploaded_file)

        if label == "fake":
            loading.empty()
            raise Exception(
                "Our algorithm concluded that you are trying to impersonate someone, please try again. ("
                "Confidence: " + str(acc) + ")")

        loading.empty()
        st.success("Authentication completed! Welcome, " + user["name"])
    except Exception as e:
        loading.empty()
        print('error', e)
        st.error(e)
