import streamlit as st
import pickle
import numpy as np

st.title("🩺 Multi-Disease Prediction System ")
st.write("This app predicts **Kidney Disease**, **Liver Disease**, and **Parkinson’s Disease** using  pickle files.")

def load_pickle(model_path, scaler_path):
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    return model, scaler


menu = st.sidebar.selectbox(
    "Disease Prediction System",
    ["Kidney Disease", "Liver Disease", "Parkinson's Disease"]
)

if menu == "Kidney Disease":
    st.header("🧪 Kidney Disease Prediction")

    # Load pickle files
    model, scaler = load_pickle("kidney_model.pkl", "kidney_scaler.pkl")

    st.subheader("Enter Patient Details")
    age = st.number_input("age", min_value=5,max_value=100,value=25)
    bp = st.number_input("bp", 10, 150, 10)
    sg = st.number_input("sg", 0.0, 2.0, 1.02)
    al = st.number_input("al", 0, 5, 1)
    su = st.number_input("su", 0, 5, 0)
    pcc = st.number_input("pcc", 0, 1, 0)
    bu = st.number_input("bu", 0, 300, 50)
    sc = st.number_input("sc", 0.0, 20.0, 1.2)
    hemo = st.number_input("hemo", 0.0, 20.0 , 1.2)
    htn = st.number_input("htn", 0, 1, 0)
    dm = st.number_input("dm", 0, 1, 0)
    appet = st.number_input("appet", 0, 1, 0)
    
    input_data = np.array([[age, bp, sg, al, su, pcc, bu, sc, hemo, htn, dm, appet]])
    scaled_data = scaler.transform(input_data)

    if st.button("Predict Kidney Disease"):
       prediction = model.predict(scaled_data)[0]
       if prediction == 1:
           st.error("⚠️ The patient has Chronic Kidney Disease(ckd)")
       else:
           st.success("🟢 The patient does NOT Chronic Kidney Disease(not ckd)")

    
elif menu == "Liver Disease":
    st.header("🫁 Liver Disease Prediction")
    
    model, scaler = load_pickle("liver_model.pkl", "liver_scaler.pkl")

    st.subheader("Enter Patient Details")
    Age = st.number_input("Age", 1, 120, 45)
    Gender = st.number_input("Gender", 0, 1, 1)
    Total_Bilirubin = st.number_input("Total_Bilirubin", 0.0, 50.0, 1.2)
    Direct_Bilirubin = st.number_input("Direct_Bilirubin", 0.0, 20.0, 0.4)
    Alkaline_Phosphotase = st.number_input("Alkaline_Phosphotase", 0, 2000, 200)
    Alamine_Aminotransferase = st.number_input("Alamine_Aminotransferase", 0, 2000, 35)
    Aspartate_Aminotransferase = st.number_input("Aspartate_Aminotransferase", 0, 2000, 30)
    Total_Protiens = st.number_input("Total_Protiens", 0.0, 10.0, 7.0)
    Albumin = st.number_input("Albumin", 0.0, 6.0, 3.5)
    Albumin_and_Globulin_Ratio = st.number_input("Albumin_and_Globulin_Ratio", 0.0, 5.0, 1.1)

    input_data = np.array([[Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]])
    scaled_data = scaler.transform(input_data)

    if st.button("Predict Liver Disease"):
        prediction = model.predict(scaled_data)[0]
        if prediction == 1:
            st.error("⚠️ Liver Patient")
        else:
            st.success("🟢 No Liver Patient")
            

elif menu == "Parkinson's Disease":
    st.header("🧠 Parkinson's Disease Prediction")

    # Load pickle files
    model, scaler = load_pickle("parkinson_model.pkl", "parkinson_scaler.pkl")

    st.subheader("Enter Voice Frequency Features")
    fo = st.number_input("MDVP:Fo(Hz)", 0.0, 300.0, 120.0)
    fhi = st.number_input("MDVP:Fhi(Hz)", 0.0, 300.0, 150.0)
    flo = st.number_input("MDVP:Flo(Hz)", 0.0, 300.0, 70.0)
    jitter = st.number_input("Jitter (%)", 0.0, 1.0, 0.005)
    rap = st.number_input("MDVP:RAP", 0.0, 1.0, 0.003)
    shimmer = st.number_input("Shimmer:APQ5", 0.0, 20.0, 0.3)
    nhr = st.number_input("NHR", 0.0, 1.0, 0.003)
    hnr = st.number_input("HNR", 0.0, 40.0, 20.0)
    rpde = st.number_input("RPDE", 0.0, 1.0, 0.1)
    dfa = st.number_input("DFA", 0.0, 1.0, 0.1)
    ppe = st.number_input("PPE", 0.0, 1.0, 0.1)


    input_data = np.array([[fo, fhi, flo, jitter, rap, shimmer, nhr, hnr, rpde, dfa, ppe]])
    scaled_data = scaler.transform(input_data)

    if st.button("Predict Parkinson's Disease"):
        prediction = model.predict(scaled_data)[0]
        if prediction == 1:
            st.error("⚠️ Parkinson's Disease Detected")
        else:
            st.success("🟢 No Parkinson's Disease Detected")

    