import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and scaler

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Heart Disease Prediction App", page_icon="❤")

# Custom CSS styling
# st.markdown("""
#     <style>
#     .main { background-color: #f0f2f6; }
#     </style>
#     """, unsafe_allow_html=True)


# import streamlit as st
# import numpy as np
# import pickle

# st.set_page_config(page_title="Heart Disease Prediction App", page_icon="❤")

# Custom CSS for full-page creamy background
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5dc; /* Creamy color */
    }
    .main {
        background-color: #f5f5dc;
    }
    .css-18e3th9 {
        background-color: #f5f5dc; /* This targets sidebar */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Introduction


# st.markdown(
#     """
#     <style>
#     .main {
#         background-color: #f5f5dc; /* Creamy color */
#         padding: 10px;
#         border-radius: 10px;
#     }
#     .stButton>button {
#         background-color: #f5f5dc; /* Match button to background */
#         border: none;
#         padding: 10px 20px;
#         font-size: 16px;
#         color: black;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

st.title("Heart Disease Prediction App")
st.write("This application predicts the likelihood of heart disease based on several health metrics. Please enter the details below.")

# Sidebar inputs
st.sidebar.header("Input Features")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.sidebar.selectbox("Chest Pain Type (0 to 3)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results (0, 1, 2)", [0, 1, 2])
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment (0, 1, 2)", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thalassemia (1 = Normal; 2 = Fixed Defect; 3 = Reversible Defect)", [1, 2, 3])

# Prediction logic
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
input_data_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)
    
    if prediction[0] == 0:
        st.success("The person does not have heart disease.")
        st.write(f"*Confidence Level:* {probability[0][0] * 100:.2f}%")
    else:
        st.warning("The person has heart disease.")
        st.write(f"*Confidence Level:* {probability[0][1] * 100:.2f}%")
        st.write("### Health Tips:")
        st.write("1. Maintain a balanced diet.")
        st.write("2. Exercise regularly.")
        st.write("3. Avoid smoking and limit alcohol intake.")
        st.write("4. Regular check-ups with a healthcare provider.")

# # Data Distribution


# # Footer
# # st.markdown("""
# # <hr>
# # <p style='text-align: center;'>
# #     Heart Disease Prediction App | Developed with ❤ by [Your Name]
# #     <br>Dataset: UCI Machine Learning Repository
# # </p>
# # """, unsafe_allow_html=True)

# import streamlit as st
# import numpy as np
# import pickle
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# st.set_page_config(page_title="Heart Disease Prediction App", page_icon="❤")

# # Custom CSS styling
# st.markdown("<h1 style='text-align: center;'> Heart Disease Prediction App </h1>", unsafe_allow_html=True)
# st.write("This application predicts the likelihood of heart disease based on several health metrics.")
# #st.write("Use the sidebar to enter the details and click 'Predict' to get the results.")

# # Sidebar inputs
# st.sidebar.header("Input Features")
# age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
# sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
# cp = st.sidebar.selectbox("Chest Pain Type (0 to 3)", [0, 1, 2, 3])
# trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
# chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
# fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
# restecg = st.sidebar.selectbox("Resting Electrocardiographic Results (0, 1, 2)", [0, 1, 2])
# thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
# exang = st.sidebar.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
# oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
# slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment (0, 1, 2)", [0, 1, 2])
# ca = st.sidebar.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
# thal = st.sidebar.selectbox("Thalassemia (1 = Normal; 2 = Fixed Defect; 3 = Reversible Defect)", [1, 2, 3])

# # Load model and scaler
# with open('model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)
# with open('scaler.pkl', 'rb') as scaler_file:
#     scaler = pickle.load(scaler_file)

# input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
# input_data_scaled = scaler.transform(input_data)

# result_placeholder = st.empty()

# if st.button("Predict"):
#     prediction = model.predict(input_data_scaled)
#     probability = model.predict_proba(input_data_scaled)
    
#     with result_placeholder:
#         st.markdown("---")
#         st.subheader("Prediction Result:")
        
#         if prediction[0] == 0:
#             st.success("The person does not have heart disease.")
#             st.write(f"*Confidence Level:* {probability[0][0] * 100:.2f}%")
#         else:
#             st.warning("The person has heart disease.")
#             st.write(f"*Confidence Level:* {probability[0][1] * 100:.2f}%")
#             st.write("### Health Tips:")
#             st.write("1. Maintain a balanced diet.")
#             st.write("2. Exercise regularly.")
#             st.write("3. Avoid smoking and limit alcohol intake.")
#             st.write("4. Regular check-ups with a healthcare provider.")