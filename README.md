# 🎓 Student Burnout Risk Classifier

A machine learning web app that predicts a student's **burnout risk level** (Low / Medium / High) based on their AI tool usage habits and academic information.

Built with a **Support Vector Classifier (SVC)** trained on the AI Student Impact Dataset, and deployed as an interactive [Streamlit](https://streamlit.io/) web application.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

> Replace the link above with your Streamlit Community Cloud URL after deployment.

---

## 📋 Features

- Predicts burnout risk as **Low**, **Medium**, or **High**
- 14 input features covering academic performance, AI usage habits, and personal indicators
- Color-coded results with actionable recommendations
- Built-in input summary panel

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | Support Vector Classifier (SVC) |
| Dataset | AI Student Impact Dataset |
| Preprocessing | StandardScaler |
| Output Classes | Low / Medium / High burnout risk |

### Input Features

| Feature | Type |
|---|---|
| Major Category | Categorical (Arts, Business, Humanities, Medical, STEM) |
| Year of Study | Ordinal (Freshman → Graduate) |
| Pre-Semester GPA | Continuous (0.0 – 4.0) |
| Post-Semester GPA | Continuous (0.0 – 4.0) |
| Weekly GenAI Usage (hrs) | Continuous (0 – 40) |
| Traditional Study Hours (hrs/week) | Continuous (0 – 40) |
| Primary AI Use Case | Categorical |
| Prompt Engineering Skill | Ordinal (Beginner → Advanced) |
| Institutional Policy on AI | Categorical |
| Number of AI Tools Used | Integer (1 – 10) |
| Perceived AI Dependency | Integer (1 – 5) |
| Exam Anxiety Level | Integer (1 – 10) |
| Skill Retention Score | Continuous (0 – 100) |
| Paid AI Subscription | Binary (Yes / No) |

---

## 🗂️ Project Structure

```
student-burnout-classifier/
│
├── app.py                                      # Streamlit web application
├── svc_model.pkl                               # Trained SVC model
├── scaler.pkl                                  # Fitted StandardScaler
├── student_burnout_classification.ipynb        # Training notebook
├── requirements.txt                            # Python dependencies
├── .streamlit/
│   └── config.toml                             # Streamlit theme & server config
├── .gitignore
└── README.md
```

---

## ⚙️ Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/student-burnout-classifier.git
cd student-burnout-classifier
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deploy on Streamlit Community Cloud (Free)

1. Push this repo to GitHub (make sure `svc_model.pkl` and `scaler.pkl` are included)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo and branch → set **Main file path** to `app.py`
4. Click **Deploy** — your app will be live in ~1 minute

---

## 📦 Dependencies

```
streamlit==1.45.1
scikit-learn==1.6.1
numpy==2.0.2
joblib==1.4.2
```

---

## 📝 License

This project was developed as part of an AI & Data Science Internship. For educational and research use.
