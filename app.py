"""
Student Burnout Risk Classifier - Streamlit App
================================================
This app loads the trained SVC model and scaler to predict
a student's burnout risk level based on their AI usage habits
and academic information.

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Burnout Risk Classifier",
    page_icon="🎓",
    layout="centered"
)

# ─── Load Model and Scaler ───────────────────────────────────────────────────
# We use st.cache_resource so the model is loaded only once
@st.cache_resource
def load_model_and_scaler():
    model  = joblib.load("svc_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# ─── Title and Description ────────────────────────────────────────────────────
st.title("🎓 Student Burnout Risk Classifier")
st.markdown("""
This application uses a **Support Vector Classifier (SVC)** trained on the
AI Student Impact Dataset to predict a student's **burnout risk level**
based on their academic habits and AI tool usage.

Fill in the information below and click **Predict** to get a result.

Built By: Noureldin Bassem Mohamed
""")
st.divider()

# ─── User Input ───────────────────────────────────────────────────────────────
st.header("📋 Student Information")

col1, col2 = st.columns(2)

with col1:
    major_category = st.selectbox(
        "Major Category",
        options=["Arts", "Business", "Humanities", "Medical", "STEM"],
        help="Select the student's academic major."
    )

    year_of_study = st.selectbox(
        "Year of Study",
        options=["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
        help="Select the student's current academic year."
    )

    pre_semester_gpa = st.slider(
        "Pre-Semester GPA",
        min_value=0.0, max_value=4.0, value=3.2, step=0.01,
        help="GPA recorded before the current semester."
    )

    post_semester_gpa = st.slider(
        "Post-Semester GPA",
        min_value=0.0, max_value=4.0, value=3.1, step=0.01,
        help="GPA recorded after the current semester."
    )

    weekly_genai_hours = st.slider(
        "Weekly GenAI Usage (hours)",
        min_value=0.0, max_value=40.0, value=8.0, step=0.5,
        help="Hours per week the student spends using Generative AI tools."
    )

    traditional_study_hours = st.slider(
        "Traditional Study Hours (per week)",
        min_value=0.0, max_value=40.0, value=12.0, step=0.5,
        help="Hours per week studying without AI assistance."
    )

with col2:
    primary_use_case = st.selectbox(
        "Primary Use Case for AI",
        options=[
            "Copywriting/Drafting",
            "Debugging/Troubleshooting",
            "Direct_Answer_Generation",
            "Ideation",
            "Summarizing_Reading"
        ],
        help="What the student mainly uses AI for."
    )

    prompt_engineering_skill = st.selectbox(
        "Prompt Engineering Skill",
        options=["Beginner", "Intermediate", "Advanced"],
        help="The student's proficiency with writing AI prompts."
    )

    institutional_policy = st.selectbox(
        "Institutional Policy on AI",
        options=["Actively_Encouraged", "Allowed_With_Citation", "Strict_Ban"],
        help="University's policy regarding AI tool usage."
    )

    tool_diversity = st.slider(
        "Number of AI Tools Used",
        min_value=1, max_value=10, value=3,
        help="How many different AI tools the student uses."
    )

    perceived_ai_dependency = st.slider(
        "Perceived AI Dependency (1–5)",
        min_value=1, max_value=5, value=3,
        help="Self-reported level of dependency on AI (1 = low, 5 = high)."
    )

    anxiety_level = st.slider(
        "Exam Anxiety Level (1–10)",
        min_value=1, max_value=10, value=5,
        help="Self-reported anxiety level during exams (1 = low, 10 = very high)."
    )

    skill_retention_score = st.slider(
        "Skill Retention Score (0–100)",
        min_value=0.0, max_value=100.0, value=75.0, step=0.5,
        help="Score measuring how well the student retains learned skills."
    )

    paid_subscription = st.radio(
        "Has a Paid AI Subscription?",
        options=["Yes", "No"],
        horizontal=True,
        help="Whether the student pays for an AI tool subscription."
    )

st.divider()

# ─── Preprocessing ───────────────────────────────────────────────────────────
# We apply the SAME encoding used during training

# Ordinal encoding
year_order = {"Freshman": 0, "Sophomore": 1, "Junior": 2, "Senior": 3, "Graduate": 4}
skill_order = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}

# Label encoding (must match training order exactly)
# Training used sklearn LabelEncoder which sorts alphabetically
major_map   = {"Arts": 0, "Business": 1, "Humanities": 2, "Medical": 3, "STEM": 4}
use_case_map = {
    "Copywriting/Drafting": 0,
    "Debugging/Troubleshooting": 1,
    "Direct_Answer_Generation": 2,
    "Ideation": 3,
    "Summarizing_Reading": 4
}
policy_map = {
    "Actively_Encouraged": 0,
    "Allowed_With_Citation": 1,
    "Strict_Ban": 2
}

# Assemble the feature vector
# Column order must match training: Major_Category, Year_of_Study, Pre_Semester_GPA,
# Weekly_GenAI_Hours, Primary_Use_Case, Prompt_Engineering_Skill, Tool_Diversity,
# Paid_Subscription, Traditional_Study_Hours, Perceived_AI_Dependency,
# Institutional_Policy, Anxiety_Level_During_Exams, Post_Semester_GPA, Skill_Retention_Score

features = np.array([[
    major_map[major_category],
    year_order[year_of_study],
    pre_semester_gpa,
    weekly_genai_hours,
    use_case_map[primary_use_case],
    skill_order[prompt_engineering_skill],
    tool_diversity,
    1 if paid_subscription == "Yes" else 0,
    traditional_study_hours,
    perceived_ai_dependency,
    policy_map[institutional_policy],
    anxiety_level,
    post_semester_gpa,
    skill_retention_score
]])

# ─── Prediction ───────────────────────────────────────────────────────────────
if st.button("🔍 Predict Burnout Risk", use_container_width=True, type="primary"):

    # Scale the input using the saved scaler
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)[0]

    # Map back to label
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    result = label_map[prediction]

    st.divider()
    st.header("🔎 Prediction Result")

    # Display result with color coding
    if result == "Low":
        st.success(f"### ✅ Burnout Risk Level: **{result}**")
        st.markdown("""
        **What this means:**  
        This student shows a **low risk** of burnout. Their AI usage habits and 
        academic performance indicators suggest a healthy balance.  
        
        **Recommendation:** Continue monitoring regularly and encourage healthy study habits.
        """)

    elif result == "Medium":
        st.warning(f"### ⚠️ Burnout Risk Level: **{result}**")
        st.markdown("""
        **What this means:**  
        This student shows a **moderate risk** of burnout. Some indicators suggest 
        potential stress factors that should be addressed.  
        
        **Recommendation:** Consider academic counseling and review AI usage patterns.
        """)

    else:  # High
        st.error(f"### 🚨 Burnout Risk Level: **{result}**")
        st.markdown("""
        **What this means:**  
        This student shows a **high risk** of burnout. Multiple risk factors are present.
        
        **Recommendation:** Immediate academic support and well-being check-in are advised.
        """)

    # Show a summary of the input
    with st.expander("📊 View Input Summary"):
        st.write("| Feature | Value |")
        st.write("|---|---|")
        st.write(f"| Major Category | {major_category} |")
        st.write(f"| Year of Study | {year_of_study} |")
        st.write(f"| Pre-Semester GPA | {pre_semester_gpa} |")
        st.write(f"| Post-Semester GPA | {post_semester_gpa} |")
        st.write(f"| Weekly GenAI Hours | {weekly_genai_hours} |")
        st.write(f"| Traditional Study Hours | {traditional_study_hours} |")
        st.write(f"| Primary Use Case | {primary_use_case} |")
        st.write(f"| Prompt Engineering Skill | {prompt_engineering_skill} |")
        st.write(f"| Tool Diversity | {tool_diversity} |")
        st.write(f"| Paid Subscription | {paid_subscription} |")
        st.write(f"| Perceived AI Dependency | {perceived_ai_dependency} |")
        st.write(f"| Institutional Policy | {institutional_policy} |")
        st.write(f"| Exam Anxiety Level | {anxiety_level} |")
        st.write(f"| Skill Retention Score | {skill_retention_score} |")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("""
🎓 Student Burnout Risk Classifier | Built with Support Vector Classifier (SVC)  
Built by: Noureldin Bassem Mohamed
""")
