import pandas as pd
import numpy as np
import streamlit as st
from sklearn import *
import pickle

df = pickle.load(open('data2.pkl','rb'))
pipe_rf = pickle.load(open('rf2.pkl','rb'))

st.title('Laptop Price Predictor')
st.header('Fill the details to predict the Laptop Price')

# company - dropdown
company = st.selectbox('Brand',df['Company'].unique())
# typename - dropdown
type = st.selectbox('Type',df['TypeName'].unique())
# Ram - dropdown
ram = st.selectbox('Ram(in GB)',[8, 16,  4,  2, 12,  6, 32, 24, 64])
# weight - number_input
weight = st.number_input('Weight of the laptop')
# Touchscreen
touchscreen = st.selectbox('TouchScreen',['No','Yes'])
# IPS
ips = st.selectbox('IPS',['No','Yes'])
# CPU
# cpu = st.selectbox('CPU',df['Cpu brand'].unique())
cpu = st.selectbox('CPU',['Intel Core i5', 'Intel Core i7', 'AMD Processor', 
                    'Intel Core i3','Other Intel Processor'])
# hdd
hdd =  st.selectbox('HDD(in GB)',[0,32,128,500,1000,2000])
# ssd
ssd =  st.selectbox('SSD(in GB)',[0,8,16,32,64,128,180,240,256,512,1000,1024])
# GPU
gpu = st.selectbox('GPU',df['Gpu brand'].unique())
#os
os = st.selectbox('OS',df['os'].unique())


if st.button('Predict Laptop Price'):
    if touchscreen=="Yes":
        touchscreen=1
    else:
        touchscreen=0
    if ips=="Yes":
        ips=1
    else:
        ips=0
    test_data = np.array([company,type,ram,weight,touchscreen,ips,cpu,hdd,ssd,gpu,os])
    test_data = test_data.reshape([1,11])

    st.success(pipe_rf.predict(test_data)[0])




