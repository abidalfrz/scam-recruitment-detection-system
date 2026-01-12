from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
import pandas as pd
from services.predictor import predictor
from services.preprocessing import preprocess
import numpy as np

class PredictionModel(BaseModel):
    title: str = Field(..., example="Marketing Intern")
    location: str = Field(..., example="US, NY, New York")
    department: str = Field(..., example="Marketing")
    salary_range: str = Field(..., example="missing")
    company_profile: str = Field(..., example="<h3>We're Food52, and we've created a groundbreaking and award-winning cooking site. We support, connect, and celebrate home cooks, and give them everything they need in one place.</h3>\r\n<p>We have a top editorial, business, and engineering team. We're focused on using technology to find new and better ways to connect people around their specific food interests, and to offer them superb, highly curated information about food and cooking. We attract the most talented home cooks and contributors in the country; we also publish well-known professionals like Mario Batali, Gwyneth Paltrow, and Danny Meyer. And we have partnerships with Whole Foods Market and Random House.</p>\r\n<p>Food52 has been named the best food website by the James Beard Foundation and IACP, and has been featured in the New York Times, NPR, Pando Daily, TechCrunch, and on the Today Show.</p>\r\n<p>We're located in Chelsea, in New York City.</p>")
    description: str = Field(..., example='<p>Food52, a fast-growing, James Beard Award-winning online food community and crowd-sourced and curated recipe hub, is currently interviewing full- and part-time unpaid interns to work in a small team of editors, executives, and developers in its New York City headquarters.</p>\r\n<ul>\r\n<li>Reproducing and/or repackaging existing Food52 content for a number of partner sites, such as Huffington Post, Yahoo, Buzzfeed, and more in their various content management systems</li>\r\n<li>Researching blogs and websites for the Provisions by Food52 Affiliate Program</li>\r\n<li>Assisting in day-to-day affiliate program support, such as screening affiliates and assisting in any affiliate inquiries</li>\r\n<li>Supporting with PR &amp; Events when needed</li>\r\n<li>Helping with office administrative work, such as filing, mailing, and preparing for meetings</li>\r\n<li>Working with developers to document bugs and suggest improvements to the site</li>\r\n<li>Supporting the marketing and executive staff</li>\r\n</ul>')
    requirements: str = Field(..., example='<ul>\r\n<li>Experience with content management systems a major plus (any blogging counts!)</li>\r\n<li>Familiar with the Food52 editorial voice and aesthetic</li>\r\n<li>Loves food, appreciates the importance of home cooking and cooking with the seasons</li>\r\n<li>Meticulous editor, perfectionist, obsessive attention to detail, maddened by typos and broken links, delighted by finding and fixing them</li>\r\n<li>Cheerful under pressure</li>\r\n<li>Excellent communication skills</li>\r\n<li>A+ multi-tasker and juggler of responsibilities big and small</li>\r\n<li>Interested in and engaged with social media like Twitter, Facebook, and Pinterest</li>\r\n<li>Loves problem-solving and collaborating to drive Food52 forward</li>\r\n<li>Thinks big picture but pitches in on the nitty gritty of running a small company (dishes, shopping, administrative support)</li>\r\n<li>Comfortable with the realities of working for a startup: being on call on evenings and weekends, and working long hours</li>\r\n</ul>')
    benefits: str = Field(..., example="missing")
    telecommuting: str = Field(..., example='f')
    has_company_logo: str = Field(..., example='t')
    has_questions: str = Field(..., example='f')
    employment_type: str = Field(..., example='Other')
    required_experience: str = Field(..., example='Internship')
    required_education: str = Field(..., example="missing")
    industry: str = Field(..., example="missing")
    function: str = Field(..., example='Marketing')
    in_balanced_dataset: str = Field(..., example='f')

class PredictionResponse(BaseModel):
    prediction: str
    probability: float

    class Config:
        from_attributes = True

app = FastAPI(title="Job Scam Detection API")

@app.get("/")
def index():
    return {"message": "Welcome to the Job Scam Detection API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionModel):
    data_dict = data.dict()
    df = pd.DataFrame([data_dict])
    X = preprocess(df)
    prediction = predictor.predict(X)[0]
    prediction_proba = predictor.predict_proba(X)[0][prediction]
    mapping = {0: "Real", 1: "Fraudulent"}
    prediction = mapping[prediction]
    response = {
        "prediction": prediction,
        "probability": float(prediction_proba)
    }
    return response

if __name__ == "__main__":
    FILE_NAME = "app"
    ENTRY_POINT = "app"
    HOST = "127.0.0.1"
    PORT = 8000
    uvicorn.run(f"{FILE_NAME}:{ENTRY_POINT}", host=HOST, port=PORT, reload=True)

