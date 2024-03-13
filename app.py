import streamlit as st
from auth import authenticate
from handlers import verifyImage
from test import test
import cv2
import numpy as np

st.markdown("<h1 style='text-align: center; '>Login</h1>", unsafe_allow_html=True)
global authenticated, user
authenticated = False

with st.form("my_form"):
   name = st.text_input("Username")
   password = st.text_input("Password", type="password")
  
   submitted = st.form_submit_button("Submit")
   if submitted:
      if(not name or not password):
        st.error("Please enter username and password")
      else:
        user = authenticate(name, password)
        if(len(user) == 0):
          st.error("Invalid credentials")
        else:
          authenticated = True

if(True):
  st.markdown("<p>Suspicious activity has been reported on this account recently, to confirm its authenticity take a photo so we can perform facial recognition.</p>", unsafe_allow_html=True)

  uploaded_file = st.camera_input(label="Your webcam", label_visibility="hidden")

  if uploaded_file is not None:
    file_buffer = bytearray(uploaded_file.getvalue())
    sendImage = st.button("Submit photo")
 
    if(sendImage):
      try:
        loading = st.empty()
        loading.write("Processing face recognition...")
        loading.write("Approximate response time: 3 minutes")
        verifyImage(file_buffer)

        cv2_img = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        liveness_test = test(
          image=cv2_img,
          model_dir="./resources/anti_spoof_models",
          device_id=0
        )

        if(liveness_test != 1):
          raise Exception("Your picture was not valid, try to take another one.")

        loading.empty()
        st.success("Your picture was valid. Your autentication is completed!")
      except Exception as e:
        loading.empty()
        print('error', e)
        st.error(e)
