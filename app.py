import streamlit as st  
import pandas as pd
import joblib
# Load the trained model
model1_column = joblib.load('columns.pkl')
model2 = joblib.load('KNN_heart.pkl')
model3_scaler = joblib.load('scalar.pkl')

st.title("Heart Disease Prediction")
st.markdown("This app predicts the likelihood of heart disease based on user input.")
age=st.slider("AGE", 20, 80, 50)
sex=st.selectbox("SEX",["Male","Female"])
cp=st.selectbox("Chest Pain Type",["Typical Angina","ATA","NAP","TA","ASY"])
resting_bp=st.slider("Resting Blood Pressure(mm Hg)", 80, 200, 120)
cholesterol=st.slider("Cholesterol(mg/dl)", 100, 600, 200)
fbs=st.selectbox("Fasting Blood Sugar > 120 mg/dl",[0,1])    
rest_ecg=st.selectbox("Resting ECG",["Normal","ST-T wave abnormality","Left ventricular hypertrophy"])
max_hr=st.slider("Maximum Heart Rate", 60, 220, 150)
exercise_angina=st.selectbox("Exercise Induced Angina",["Yes","No"])
oldpeak=st.slider("Oldpeak", 0.0, 6.0, 1.0)
st_slope=st.selectbox("ST Slope",["Up","Flat","Down"])


if st.button("Predict"):
     raw_input={
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBloodSugar': fbs,
        'RestingECG': rest_ecg,
        'MaximumHeartRate': max_hr,
        'ExerciseInducedAngina_'+ exercise_angina: 1,
        'Oldpeak': oldpeak,
        'ST_Slope_'+ st_slope: 1,
        'Sex_'+ sex: 1,
        'ChestPainType_'+ cp: 1,
    }
     
     input_df = pd.DataFrame([raw_input])

     for col in model1_column:
        if col not in input_df.columns:
          input_df[col] = 0

     input_df = input_df[model1_column]

     scaled_input = model3_scaler.transform(input_df)


     prediction = model2.predict(scaled_input)[0]

     if prediction == 1:
        st.error("The model predicts that you are likely to have heart disease. Please consult a healthcare professional for further evaluation.")
     else:   
       st.success("The model predicts that you are unlikely to have heart disease. However, please remember that this is just a prediction and not a diagnosis. Always consult a healthcare professional for any health concerns.")  
