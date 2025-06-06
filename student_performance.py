import streamlit as st 
import pandas as pd 
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load model and preprocessing tools
def load_model():
    with open("student_lr_final_model.pkl", "rb") as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

# Preprocess user input
def preprocessing_input_data(data, scaler, le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

# Run prediction
def predict_data(data):
    model, scaler, le = load_model()
    processed_data = preprocessing_input_data(data, scaler, le)
    prediction = model.predict(processed_data)
    return prediction

# Main app
def main():
    st.title("🎓 Student Performance Prediction")
    st.write("Enter your data to get a prediction of your performance.")

    # Input fields
    hour_studied = st.number_input("📚 Hours Studied", min_value=1, max_value=10, value=5)
    previous_score = st.number_input("📝 Previous Score", min_value=40, max_value=100, value=70)
    extra = st.selectbox("🏆 Extracurricular Activity", ["Yes", "No"])
    sleeping_hour = st.number_input("😴 Sleep Hours", min_value=4, max_value=10, value=7)
    number_of_paper_solved = st.number_input("📄 Question Papers Solved", min_value=0, max_value=10, value=5)

    # Prediction
    if st.button("🔍 Predict Your Score"):
        user_data = {
            "Hours Studied": hour_studied,
            "Previous Scores": previous_score,
            "Extracurricular Activities": extra,
            "Sleep Hours": sleeping_hour,
            "Sample Question Papers Practiced": number_of_paper_solved
        }
        prediction = predict_data(user_data)
        st.success(f"🎯 Predicted Performance Category: **{prediction[0]}**")

    # Footer
    st.markdown("---", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center;'>
            <strong>Made by <a href='https://www.linkedin.com/in/rajat-kumar-sahu1/' target='_blank'>Rajat Kumar Sahu</a></strong><br>
            📧 <a href='mailto:rajatks1997@gmail.com'>rajatks1997@gmail.com</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Run the app
if __name__ == "__main__":
    main()
