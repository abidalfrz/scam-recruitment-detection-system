# 🛡️ Job Recruitment Scam Detection

This repository contains a natural language processing (NLP) project designed to detect job recruitment scams. This project leveraging deep learning-based embedding, statistical feature engineering, and ensemble learning through Voting Classifier to classify job postings as legitimate or fraudulent. Utilize FastAPI for building the API server and Docker for containerization, ensuring easy deployment and scalability.

---

## 📌 Problem Statement

Online recruitment fraud is a rising cybersecurity threat. Scammers post fake job advertisements to steal personal information, money, or compromise user security. Distinguishing these scams from real jobs is increasingly difficult as scammers use sophisticated language to mimic legitimate companies.

This project aims to:
- **Develop a robust NLP model** that can accurately classify job postings as legitimate or scam.
- **Prevent financial and data loss** for job seekers by identifying fraudulent job listings.
- **Facilitate safer job search experiences** through automated scam detection.

---

## 📊 Features (Dataset)

The dataset contains the following features:

| Feature Name                   | Description                                         | Type         |
|-------------------------------|-----------------------------------------------------|--------------|
| `title`                       | Job title                                          | Categorical  |
| `location`                    | Job location                                       | Categorical  |
| `department`                  | Department of the job                              | Categorical  |
| `salary_range`                | Salary range offered                               | Categorical  |
| `company_profile`             | Profile of the company                             | Text         |
| `description`                 | Job description                                    | Text         |
| `requirements`                | Job requirements                                   | Text         |
| `benefits`                    | Benefits offered                                   | Text         |
| `telecommuting`               | Whether telecommuting is allowed                   | Binary (t/f) |
| `has_company_logo`            | Whether the job posting has a company logo         | Binary (t/f) |
| `has_questions`               | Whether the job posting has questions               | Binary (t/f) |
| `employment_type`             | Type of employment                                 | Categorical  |
| `required_experience`         | Required experience level                          | Categorical  |
| `required_education`          | Required education level                           | Categorical  |
| `industry`                    | Industry of the job                                | Categorical  |
| `function`                    | Job function                                      | Categorical  |
| `in_balanced_dataset`         | Whether the job posting is in a balanced dataset    | Binary (t/f) |
| `fraudulent`                  | Target variable indicating if the job is a scam    | Binary (0/1) |

---

## 🛠️ Tech Stack

### Backend:
- **Language:** Python
- **Framework:** FastAPI
- **ASGI Server:** Uvicorn
- **Validation:** Pydantic

### Data Science & ML:
- **Data Handling:** Pandas
- **Numerical Computing:** NumPy
- **Data Visualization:** Matplotlib, Seaborn, WordCloud
- **Text Processing:** NLTK, SentenceTransformers (`paraphrase-multilingual-mpnet-base-v2`)
- **Machine Learning Algorithms:** scikit-learn, XGBoost, LightGBM

### DevOps & Experiments:
- **Containerization:** Docker & Docker Desktop
- **Experimentation:** Jupyter Notebooks

---

## 📁 Project Structure

```bash
scam-recruitment-detection-system/
│
├── artifacts/             # Trained models and preprocessors
│   ├── best_model.pkl     # The trained classifier
│   ├── encoder.pkl        # Categorical encoder
│   └── scaler_final.pkl   # Numerical scaler
│
├── data/
│   └── data.csv           # Dataset file
│
├── notebooks/
│   └── main.ipynb         # Main experimentation notebook
│
├── services/              # Application logic modules
│   ├── __init__.py
│   ├── predictor.py       # Logic to load model & predict
│   └── preprocessing.py   # Text cleaning & Feature Engineering
│
├── app.py                 # Main application entry point
├── Dockerfile             # Docker image configuration
├── docker-compose.yml     # Docker compose configuration
├── requirements.txt       # Python dependencies list
├── .gitignore
├── README.md
└── .dockerignore
```

---

## 🔁 Workflow

1. **Data Collection:** The dataset contains job postings labeled as legitimate or scam and collected from kaggle.
2. **Exploratory Data Analysis (EDA):** Analyze the dataset to understand feature distributions, identify missing values, and visualize text data using word clouds and other plots conducted in `main.ipynb`.
3. **Data Preprocessing:** Clean and preprocess text features, handle missing values, encode categorical variables, and performs embedding and feature engineering conducted in `preprocessing.py`.
4. **Model Training:** Experiments with different algorithms in `model.ipynb`, leading to the selection of a Voting Classifier, combining SVM, XGBoost, and LightGBM for optimal performance.
5. **Model Evaluation:** Evaluate the model using ROC-AUC Score and Weighted F1-Score to ensure robustness against class imbalance.
6. **API Development:** Build a FastAPI application in `app.py` to serve the model for real-time predictions.
7. **Containerization:** Use Docker to containerize the application for easy deployment.

---

## 📂 Dataset & Credits

The dataset used in this project was sourced from Kaggle.  
You can access the original dataset and description through the link below:

🔗[Job Scam Recruitment Dataset](https://www.kaggle.com/datasets/amruthjithrajvr/recruitment-scam)

We would like to acknowledge and thanks to the dataset creator for making this resource publicly available for research and educational use.

---

## 🚀 How to Run

## Clone the Repository

```bash
git clone https://github.com/abidalfrz/scam-recruitment-detection-system.git
cd scam-recruitment-detection-system
```

## Option 1: Manual Setup

### 1. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Linux/macOS
venv\Scripts\activate.bat     # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py

# The API will be accessible at http://localhost:8000
```

## Option 2: Using Docker
Make sure you have Docker installed on your machine.

### 1. Build and Run with Docker Compose

```bash
docker-compose up --build
```

This command will build the Docker image based on `Dockerfile` and start the FastAPI application in a container.

## Access the API

Open your web browser and navigate to http://localhost:8000/docs to access the interactive API documentation provided by Swagger UI. You can test the endpoints directly from this interface.

---








