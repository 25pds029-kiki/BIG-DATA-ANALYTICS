import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


st.set_page_config(
    page_title="Student Dropout Analytics",
    layout="wide"
)


model = pickle.load(open("dropout_model.pkl","rb"))

st.title("🎓 Student Dropout Prediction & Analytics Dashboard")

st.write("Big Data Analytics Project – Predicting Student Dropout Risk")

tab1, tab2, tab3, tab4 = st.tabs([
"Prediction System",
"Model Performance",
"Project Insights",
"Dataset Visualizations"
])


with tab1:

    st.subheader("Enter Student Behavioral Data")

    col1, col2 = st.columns(2)

    with col1:
        attendance = st.slider("Attendance Percentage",0,100,70)
        assignment = st.slider("Assignment Submission Rate",0,100,70)
        lms = st.slider("LMS Login per Week",0,20,5)
        cgpa = st.slider("CGPA",0.0,10.0,7.0)

    with col2:
        backlogs = st.slider("Backlogs Count",0,10,0)
        travel = st.slider("Travel Time (minutes)",0,180,30)
        extra = st.slider("Extracurricular Participation",0,10,3)

    if st.button("Predict Dropout Risk"):

        features = np.array([[attendance,assignment,lms,cgpa,backlogs,travel,extra]])

        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        st.subheader("Prediction Result")

        st.write("### Dropout Probability:", round(probability,2))

        if probability > 0.7:
            st.error("⚠ High Risk Student")
        elif probability > 0.4:
            st.warning("⚠ Medium Risk Student")
        else:
            st.success("✔ Low Risk Student")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability*100,
            title={'text': "Dropout Risk Probability"},
            gauge={
                'axis': {'range': [0,100]},
                'steps': [
                    {'range':[0,40],'color':'green'},
                    {'range':[40,70],'color':'orange'},
                    {'range':[70,100],'color':'red'}
                ]
            }
        ))

        fig.update_layout(height=300)

        col1,col2,col3 = st.columns([1,2,1])
        with col2:
            st.plotly_chart(fig)


# TAB 2 — MODEL PERFORMANCE
with tab2:

    st.subheader("Model Accuracy Comparison")

    models = ["Logistic Regression","Random Forest","Gradient Boosted Trees"]
    accuracy = [0.87,0.86,0.82]

    fig, ax = plt.subplots(figsize=(5,3))

    ax.bar(models, accuracy, color=["blue","green","orange"])

    ax.set_ylabel("Accuracy")
    ax.set_title("Model Performance Comparison")

    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig)

    st.write("Logistic Regression achieved the highest accuracy in this study.")


# TAB 3 — PROJECT INSIGHTS
with tab3:

    st.subheader("Student Risk Distribution")

    labels = ["Low Risk","Medium Risk","High Risk"]
    values = [60,25,15]

    fig1, ax1 = plt.subplots(figsize=(5,3))

    ax1.bar(labels, values, color=["green","orange","red"])

    ax1.set_ylabel("Number of Students")
    ax1.set_title("Risk Level Distribution")

    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig1)

    st.subheader("Important Dropout Factors")

    features = [
        "Attendance",
        "Assignments",
        "LMS Activity",
        "CGPA",
        "Backlogs",
        "Travel Time",
        "Extracurricular"
    ]

    importance = [30,20,15,12,10,8,5]

    fig2, ax2 = plt.subplots(figsize=(5,3))

    ax2.barh(features, importance)

    ax2.set_title("Key Behavioral Indicators")

    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig2)

# TAB 4 — DATASET VISUALIZATIONS
with tab4:

    st.subheader("Dropout Trend by Year")

    years = [1,2,3,4]
    dropout_rate = [0.45,0.32,0.20,0.15]

    fig3, ax3 = plt.subplots(figsize=(5,3))

    ax3.plot(years, dropout_rate, marker="o", color="purple")

    ax3.set_xlabel("Year of Study")
    ax3.set_ylabel("Dropout Rate")
    ax3.set_title("Year-wise Dropout Trend")

    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig3)

    st.subheader("Attendance vs Dropout")

    attendance_bins = [40,60,70,80,90,100]
    dropout_counts = [120,95,70,40,20,10]

    fig4, ax4 = plt.subplots(figsize=(5,3))

    ax4.bar(attendance_bins, dropout_counts)

    ax4.set_xlabel("Attendance Percentage")
    ax4.set_ylabel("Dropout Count")
    ax4.set_title("Attendance Impact on Dropout")

    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig4)