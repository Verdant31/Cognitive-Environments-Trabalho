import boto3
import streamlit as st

session = boto3.Session(aws_access_key_id=st.secrets["ACCESS_KEY"], aws_secret_access_key= st.secrets["ACCESS_SECRET"])
client = session.client('rekognition', region_name=st.secrets["REGION"])