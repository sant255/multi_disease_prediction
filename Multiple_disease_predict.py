import streamlit as st
import numpy as np
import pickle

# ---------------- Load Models & Scalers ----------------
kidney_model = pickle.load(open("kidney_lr_model.pkl", "rb"))
kidney_scaler = pickle.load(open("kidney_scaler.pkl", "rb"))

liver_model = pickle.load(open("liver_rf_smote.pkl", "rb"))
liver_scaler = pickle.load(open("liver_scaler.pkl", "rb"))

parkinsons_model = pickle.load(open("parkinsons_rf_smote.pkl", "rb"))
parkinsons_scaler = pickle.load(open("parkinsons_scaler.pkl", "rb"))

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Multi Disease Prediction", layout="wide")
st.title("🩺 Multi Disease Prediction System")
st.markdown("### Kidney | Liver | Parkinson’s")

disease = st.sidebar.selectbox(
    "Select Disease",
    ["Kidney Disease", "Liver Disease", "Parkinson’s Disease"]
)

# ======================================================
# 🩺 KIDNEY DISEASE (ALL FEATURES)
# ======================================================
if disease == "Kidney Disease":
    st.header("🩺 Kidney Disease Prediction")

    id = st.number_input("id")
    age = st.number_input("Age", 1, 100)
    bp = st.number_input("Blood Pressure")
    sg = st.number_input("Specific Gravity")
    al = st.number_input("Albumin")
    su = st.number_input("Sugar")
    rbc = st.number_input("Red Blood Cells")
    pc = st.number_input("Pus Cell")
    pcc = st.number_input("Pus Cell Clumps")
    ba = st.number_input("Bacteria")
    bgr = st.number_input("Blood Glucose Random")
    bu = st.number_input("Blood Urea")
    sc = st.number_input("Serum Creatinine")
    sod = st.number_input("Sodium")
    pot = st.number_input("Potassium")
    hemo = st.number_input("Hemoglobin")
    pcv = st.number_input("Packed Cell Volume")
    wc = st.number_input("White Blood Cell Count")
    rc = st.number_input("Red Blood Cell Count")
    htn = st.number_input("Hypertension")
    dm = st.number_input("Diabetes Mellitus")
    cad = st.number_input("Coronary Artery Disease")
    appet = st.number_input("Appetite")
    pe = st.number_input("Pedal Edema")
    ane = st.number_input("Anemia")

    if st.button("Predict Kidney Disease"):
        data = np.array([[id, age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc,
                          sod, pot, hemo, pcv, wc, rc, htn, dm, cad,
                          appet, pe, ane]])

        data = kidney_scaler.transform(data)
        prediction = kidney_model.predict(data)

        if prediction[0] == 1:
            st.error("⚠️ Chronic Kidney Disease Detected")
        else:
            st.success("✅ No Kidney Disease Detected")

# ======================================================
# 🧪 LIVER DISEASE (ALL FEATURES)
# ======================================================
elif disease == "Liver Disease":
    st.header("🧪 Liver Disease Prediction")

    age = st.number_input("Age")
    gender = st.selectbox("Gender", ["Male", "Female"])
    tb = st.number_input("Total Bilirubin")
    db = st.number_input("Direct Bilirubin")
    alkphos = st.number_input("Alkaline Phosphotase")
    sgpt = st.number_input("SGPT")
    sgot = st.number_input("SGOT")
    tp = st.number_input("Total Proteins")
    alb = st.number_input("Albumin")
    ag = st.number_input("Albumin and Globulin Ratio")

    gender = 1 if gender == "Male" else 0

    if st.button("Predict Liver Disease"):
        data = np.array([[age, gender, tb, db, alkphos,
                          sgpt, sgot, tp, alb, ag]])

        data = liver_scaler.transform(data)
        prediction = liver_model.predict(data)

        if prediction[0] == 1:
            st.error("⚠️ Liver Disease Detected")
        else:
            st.success("✅ No Liver Disease Detected")

# ======================================================
# 🧠 PARKINSON’S DISEASE (ALL FEATURES)
# ======================================================
elif disease == "Parkinson’s Disease":
    st.header("🧠 Parkinson’s Disease Prediction")

    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    flo = st.number_input("MDVP:Flo(Hz)")
    jitter = st.number_input("Jitter(%)")
    jitter_abs = st.number_input("Jitter(Abs)")
    rap = st.number_input("RAP")
    ppq = st.number_input("PPQ")
    ddp = st.number_input("DDP")
    shimmer = st.number_input("Shimmer")
    shimmer_db = st.number_input("Shimmer(dB)")
    apq3 = st.number_input("APQ3")
    apq5 = st.number_input("APQ5")
    apq = st.number_input("APQ")
    dda = st.number_input("DDA")
    nhr = st.number_input("NHR")
    hnr = st.number_input("HNR")
    rpde = st.number_input("RPDE")
    dfa = st.number_input("DFA")
    spread1 = st.number_input("Spread1")
    spread2 = st.number_input("Spread2")
    d2 = st.number_input("D2")
    ppe = st.number_input("PPE")

    if st.button("Predict Parkinson’s Disease"):
        data = np.array([[fo, fhi, flo, jitter, jitter_abs, rap, ppq, ddp,
                          shimmer, shimmer_db, apq3, apq5, apq, dda,
                          nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])

        data = parkinsons_scaler.transform(data)
        prediction = parkinsons_model.predict(data)

        if prediction[0] == 1:
            st.error("⚠️ Parkinson’s Disease Detected")
        else:
            st.success("✅ No Parkinson’s Disease Detected")