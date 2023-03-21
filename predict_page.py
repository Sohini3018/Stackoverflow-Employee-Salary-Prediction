import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")
    
    Country = ("United States of America",
               "Other ",
               "United Kingdom of Great Britain and Northern Ireland",
               "Germany",
               "India",
               "Canada",
               "France",
               "Australia",
               "Sweden",
               "Netherlands",
               "Israel") 
    
    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    Country = st.selectbox("Country", Country)
    education = st.selectbox("Education Level", education)

    expericence = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        x = np.array([[Country, education, expericence ]])
        x[:,0] = le_country.fit_transform(x[:,0])
        x[:,1] = le_education.fit_transform(x[:,1])
        x = x.astype(float)

        salary = regressor.predict(x)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
