import streamlit as st
import pandas as pd
import numpy as np
from  sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pickle
import tensorflow as tf

model=tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encodeer_gender = pickle.load(file)

with open('Label_encoder_geo.pkl', 'rb') as file:
    oneHotEncoder_Geography = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Streamlit app
st.title('Customer Churn Prediction')

geography=st.selectbox('Select Geography',['France','Germany','Spain','UK'])
gender=st.selectbox('Select Gender',['Male','Female'])
age=st.slider('Select Age',18,92)
balance=st.number_input('Select Balance',0,10000000)
credit_score=st.number_input('Select Credit Score',0,1000)
estimation_salary=st.number_input('Select Estimated Salary',0,10000000)
tenure=st.slider('Select Tenure',1,10)
num_of_products=st.slider('Select Number of Products',1,10)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encodeer_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimation_salary],   
})

geo_encoded=oneHotEncoder_Geography.transform([[geography]])
geo_enoded_df=pd.DataFrame(geo_encoded,columns=oneHotEncoder_Geography.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_enoded_df],axis=1)

scaled_input_data=scaler.transform(input_data)

prediction=model.predict(scaled_input_data)
prediction=prediction[0][0]

st.write(f'Churn probability is: {prediction:.2f}')

if prediction>0.5:
    st.write('The customer is likey toChurn')
else:
    st.write('The customer is not likely toChurn')








