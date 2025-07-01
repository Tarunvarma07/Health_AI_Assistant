import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import requests
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
HF_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Example model

API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Hugging Face Text Generation Function
@st.cache_data(show_spinner=False)
def query_huggingface(prompt):
    payload = {"inputs": prompt, "parameters": {"temperature": 0.7, "max_new_tokens": 512}}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            return response.json()[0]['generated_text']
        except:
            return "Error: Could not parse Hugging Face response."
    else:
        return f"Error {response.status_code}: {response.text}"

# Patient Query Function 🩺
@st.cache_data(show_spinner=False)
def answer_patient_query(query):
    query_prompt = f"""
You are a healthcare AI assistant. Respond clearly and compassionately to the following patient question:
"{query}"

Only include the response. Do not include any headings or formatting instructions.
"""
    answer = query_huggingface(query_prompt)
    if len(answer.strip()) < 50:
        answer += "\n\nIt is recommended to consult a healthcare professional for further guidance."
    return answer

# Disease Prediction Function 🔬
@st.cache_data(show_spinner=False)
def predict_disease(symptoms, age, gender, medical_history, avg_heart_rate, avg_bp_systolic, avg_bp_diastolic, avg_glucose, recent_symptoms):
    prediction_prompt = f"""
You are a medical AI assistant. Predict potential health conditions for a patient with the following data:

Symptoms: {symptoms}
Age: {age}
Gender: {gender}
Medical History: {medical_history}
Vital Signs:
- Avg Heart Rate: {avg_heart_rate} bpm
- Avg BP: {avg_bp_systolic}/{avg_bp_diastolic} mmHg
- Avg Glucose: {avg_glucose} mg/dL
- Recent Symptoms: {recent_symptoms}

Provide the top 3 possible conditions with a brief explanation and recommended next steps.
Only include the results.
"""
    return query_huggingface(prediction_prompt)

# Treatment Plan Function 💊
@st.cache_data(show_spinner=False)
def generate_treatment_plan(condition, age, gender, medical_history):
    treatment_prompt = f"""
You are a healthcare AI assistant. Create a treatment plan for the following patient:

- Condition: {condition}
- Age: {age}
- Gender: {gender}
- Medical History: {medical_history}

Provide:
1. Recommended medications
2. Lifestyle changes
3. Follow-up tests
4. Dietary and activity suggestions

Only include the treatment plan without extra headings or instructions.
"""
    return query_huggingface(treatment_prompt)

# --- UI Code Starts Here ---

st.set_page_config(page_title='HealthAI - Intelligent Healthcare Assistant', page_icon='🩺', layout='wide')
st.markdown('<style>body {background-color: #f6fafd;}</style>', unsafe_allow_html=True)

# Sidebar: Patient Profile
st.sidebar.header('Patient Profile')
name = st.sidebar.text_input('Name', value='')
age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=30)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Other'])
medical_history = st.sidebar.text_area('Medical History', value='')
current_medications = st.sidebar.text_area('Current Medications', value='')
allergies = st.sidebar.text_input('Allergies', value='')

# Generate sample data
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=90)
    heart_rate = np.random.normal(74, 5, 90)
    bp_systolic = np.random.normal(121, 8, 90)
    bp_diastolic = np.random.normal(80, 5, 90)
    glucose = np.random.normal(101, 15, 90)
    symptoms = np.random.choice(['None', 'Headache', 'Nausea', 'Chest Pain', 'Dizziness', 'Fatigue'], 90, p=[0.3,0.2,0.1,0.15,0.1,0.15])
    sleep = np.random.normal(7, 1, 90)
    df = pd.DataFrame({
        'date': dates,
        'heart_rate': heart_rate,
        'bp_systolic': bp_systolic,
        'bp_diastolic': bp_diastolic,
        'glucose': glucose,
        'symptom': symptoms,
        'sleep': sleep
    })
    return df

df = generate_sample_data()

menu = st.sidebar.radio('Navigate', ['🩺 Patient Chat', '🔬 Disease Prediction', '💊 Treatment Plans', '📊 Health Analytics'])

if menu == '🩺 Patient Chat':
    st.header('🩺 24/7 Patient Support')
    st.write('Ask any health-related question for immediate assistance.')
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    user_input = st.text_input('Ask your health question...')
    if st.button('Send') and user_input:
        st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
        with st.spinner('AI is responding...'):
            ai_response = answer_patient_query(user_input)
        st.session_state['chat_history'].append({'role': 'ai', 'content': ai_response})
    for msg in st.session_state['chat_history']:
        if msg['role'] == 'user':
            st.info(msg['content'])
        else:
            st.success(msg['content'])

elif menu == '🔬 Disease Prediction':
    st.header('🔬 Disease Prediction System')
    st.write('Enter symptoms and patient data to receive potential condition predictions.')
    symptoms = st.text_area('Current Symptoms', placeholder='Describe symptoms in detail...')
    avg_heart_rate = round(df['heart_rate'].mean(), 1)
    avg_bp_systolic = round(df['bp_systolic'].mean(), 1)
    avg_bp_diastolic = round(df['bp_diastolic'].mean(), 1)
    avg_glucose = round(df['glucose'].mean(), 1)
    recent_symptoms = ', '.join(df['symptom'].tail(7).unique())
    if st.button('Generate Prediction') and symptoms:
        with st.spinner('AI is analyzing your symptoms...'):
            prediction = predict_disease(symptoms, age, gender, medical_history, avg_heart_rate, avg_bp_systolic, avg_bp_diastolic, avg_glucose, recent_symptoms)
        st.subheader('Potential Conditions')
        st.write(prediction)

elif menu == '💊 Treatment Plans':
    st.header('💊 Personalized Treatment Plan Generator')
    st.write('Generate customized treatment recommendations based on specific conditions.')
    condition = st.text_input('Medical Condition', value='')
    if st.button('Generate Treatment Plan') and condition:
        with st.spinner('AI is generating your treatment plan...'):
            plan = generate_treatment_plan(condition, age, gender, medical_history)
        st.subheader('Personalized Treatment Plan')
        st.write(plan)

elif menu == '📊 Health Analytics':
    st.header('📊 Health Analytics Dashboard')
    st.write('Visualize and analyze patient health data trends based on AI predictions.')

    fig_hr = go.Figure()
    fig_hr.add_trace(go.Scatter(x=df['date'], y=df['heart_rate'], mode='lines', name='Heart Rate'))
    fig_hr.update_layout(title='Heart Rate Trend (90-Day)', xaxis_title='Date', yaxis_title='Heart Rate (bpm)')

    fig_bp = go.Figure()
    fig_bp.add_trace(go.Scatter(x=df['date'], y=df['bp_systolic'], mode='lines', name='Systolic'))
    fig_bp.add_trace(go.Scatter(x=df['date'], y=df['bp_diastolic'], mode='lines', name='Diastolic'))
    fig_bp.update_layout(title='Blood Pressure Trend (90-Day)', xaxis_title='Date', yaxis_title='Blood Pressure (mmHg)')

    fig_glucose = go.Figure()
    fig_glucose.add_trace(go.Scatter(x=df['date'], y=df['glucose'], mode='lines', name='Blood Glucose'))
    fig_glucose.add_trace(go.Scatter(x=df['date'], y=[125]*len(df), mode='lines', name='Reference', line=dict(dash='dash', color='red')))
    fig_glucose.update_layout(title='Blood Glucose Trend (90-Day)', xaxis_title='Date', yaxis_title='Blood Glucose (mg/dL)')

    symptom_counts = df['symptom'].value_counts()
    fig_symptom = go.Figure(data=[go.Pie(labels=symptom_counts.index, values=symptom_counts.values)])
    fig_symptom.update_layout(title='Symptom Frequency (90-Day)')

    st.subheader('Health Metrics Summary')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Avg. Heart Rate', f"{df['heart_rate'].mean():.1f} bpm", f"{df['heart_rate'].iloc[-1] - df['heart_rate'].iloc[0]:+.1f}")
    col2.metric('Avg. Blood Pressure', f"{df['bp_systolic'].mean():.1f}/{df['bp_diastolic'].mean():.1f}", f"{df['bp_systolic'].iloc[-1] - df['bp_systolic'].iloc[0]:+.1f}")
    col3.metric('Avg. Blood Glucose', f"{df['glucose'].mean():.1f} mg/dL", f"{df['glucose'].iloc[-1] - df['glucose'].iloc[0]:+.1f}")
    col4.metric('Avg. Sleep', f"{df['sleep'].mean():.1f} hours", f"{df['sleep'].iloc[-1] - df['sleep'].iloc[0]:+.1f}")

    st.plotly_chart(fig_hr, use_container_width=True)
    st.plotly_chart(fig_bp, use_container_width=True)
    st.plotly_chart(fig_glucose, use_container_width=True)
    st.plotly_chart(fig_symptom, use_container_width=True)

    st.caption('This is an AI prediction. You must consult a doctor for professional medical advice.')


