import os
import pickle
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../artifacts/best_model.pkl')

class Predictor:
    def __init__(self):
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
predictor = Predictor()
if __name__ == "__main__":
    df = pd.DataFrame({'title': 'Marketing Intern',
            'location': 'US, NY, New York',
            'department': 'Marketing',
            'salary_range': "missing",
            'company_profile': "<h3>We're Food52, and we've created a groundbreaking and award-winning cooking site. We support, connect, and celebrate home cooks, and give them everything they need in one place.</h3>\r\n<p>We have a top editorial, business, and engineering team. We're focused on using technology to find new and better ways to connect people around their specific food interests, and to offer them superb, highly curated information about food and cooking. We attract the most talented home cooks and contributors in the country; we also publish well-known professionals like Mario Batali, Gwyneth Paltrow, and Danny Meyer. And we have partnerships with Whole Foods Market and Random House.</p>\r\n<p>Food52 has been named the best food website by the James Beard Foundation and IACP, and has been featured in the New York Times, NPR, Pando Daily, TechCrunch, and on the Today Show.</p>\r\n<p>We're located in Chelsea, in New York City.</p>",
            'description': '<p>Food52, a fast-growing, James Beard Award-winning online food community and crowd-sourced and curated recipe hub, is currently interviewing full- and part-time unpaid interns to work in a small team of editors, executives, and developers in its New York City headquarters.</p>\r\n<ul>\r\n<li>Reproducing and/or repackaging existing Food52 content for a number of partner sites, such as Huffington Post, Yahoo, Buzzfeed, and more in their various content management systems</li>\r\n<li>Researching blogs and websites for the Provisions by Food52 Affiliate Program</li>\r\n<li>Assisting in day-to-day affiliate program support, such as screening affiliates and assisting in any affiliate inquiries</li>\r\n<li>Supporting with PR &amp; Events when needed</li>\r\n<li>Helping with office administrative work, such as filing, mailing, and preparing for meetings</li>\r\n<li>Working with developers to document bugs and suggest improvements to the site</li>\r\n<li>Supporting the marketing and executive staff</li>\r\n</ul>',
            'requirements': '<ul>\r\n<li>Experience with content management systems a major plus (any blogging counts!)</li>\r\n<li>Familiar with the Food52 editorial voice and aesthetic</li>\r\n<li>Loves food, appreciates the importance of home cooking and cooking with the seasons</li>\r\n<li>Meticulous editor, perfectionist, obsessive attention to detail, maddened by typos and broken links, delighted by finding and fixing them</li>\r\n<li>Cheerful under pressure</li>\r\n<li>Excellent communication skills</li>\r\n<li>A+ multi-tasker and juggler of responsibilities big and small</li>\r\n<li>Interested in and engaged with social media like Twitter, Facebook, and Pinterest</li>\r\n<li>Loves problem-solving and collaborating to drive Food52 forward</li>\r\n<li>Thinks big picture but pitches in on the nitty gritty of running a small company (dishes, shopping, administrative support)</li>\r\n<li>Comfortable with the realities of working for a startup: being on call on evenings and weekends, and working long hours</li>\r\n</ul>',
            'benefits': "missing",
            'telecommuting': 'f',
            'has_company_logo': 't',
            'has_questions': 'f',
            'employment_type': 'Other',
            'required_experience': 'Internship',
            'required_education': "missing",
            'industry': "missing",
            'function': 'Marketing',
            'in_balanced_dataset': 'f',}, index=[0])

    X = preprocess(df)
    prediction = predictor.predict(X)[0]
    prediction_proba = predictor.predict_proba(X)
    print("Prediction:", prediction)
    print("Prediction Probability:", prediction_proba[0][prediction])