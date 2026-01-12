import os
import pickle
import re
import string
import unicodedata
import numpy as np
import pandas as pd
import ftfy
import emoji
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize

SCALER_PATH = os.path.join(os.path.dirname(__file__), '../artifacts/scaler_final.pkl')  
ENCODER_PATH = os.path.join(os.path.dirname(__file__), '../artifacts/encoder.pkl')

emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002500-\U00002BEF"  # chinese char
                        u"\U00002702-\U000027B0"
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        u"\U0001f926-\U0001f937"
                        u"\U00010000-\U0010ffff"
                        u"\u2640-\u2642"
                        u"\u2600-\u2B55"
                        u"\u200d"
                        u"\u23cf"
                        u"\u23e9"
                        u"\u231a"
                        u"\ufe0f"  # dingbats
                        u"\u3030"
                        u"\u00ae" # trade Marks ®
                        u"\u00A9" # copy Right ©
                        u"\u2122" # Trade Mark TM
                        u"\u200b"
                        u"\uf0b7"
                        "]+", flags=re.UNICODE)

def demojize_text(text):

    if emoji_pattern.search(text):
        demojize_text = emoji.demojize(text, delimiters=(" ", " "))
        return re.sub(r'\s+', ' ', demojize_text).strip()
    else:
        return text
    
def delete_emoji(text):
    return emoji_pattern.sub('', text)

def normalize_unicode(text):
    text = ftfy.fix_text(text)
    return "".join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )
    
def remove_hyphens(text):
    pattern = re.compile(r'\b(\w+)[-—‑–](\w+)\b')  

    text = pattern.sub(r'\1 \2', text)
    
    return text

def remove_currency_symbols(text):
    pattern = re.compile(r'[$€¥₹£¢₽₩₪₴₱₨฿₦₮₲₭₵₿]')  

    text = pattern.sub(' ', text)
    
    return text

def remove_bullet_points(text):
    pattern = re.compile(r'\s*[\u2022\u2023\u25E6]\s*')  

    text = pattern.sub(' ', text)
    
    return text

def cleaned_text(text):
    text = normalize_unicode(text)
    text = demojize_text(text)
    text = delete_emoji(text)

    text = text.lower()
    text = re.sub(r'[\t\r]', ' ', text)
    text = re.sub(r'<b>', ' ', text)  
    text = re.sub(r'</b>', ' ', text)  
    text = re.sub(r'<i>', ' ', text)
    text = re.sub(r'</i>', ' ', text)
    text = re.sub(r'<br>', ' ', text)
    text = re.sub(r'([^.?!])\s*</br>', r'\1.</br>', text)
    text = re.sub(r'([^.?!])\s*</p>', r'\1.</p>', text)
    text = re.sub(r'\s*=\s*', '=', text)
    text = re.sub(r'<a href=.*?>', '', text)
    text = re.sub(r'</a>', '', text)
    text = re.sub(r'\(#URL.*?#\)', '', text)
    text = re.sub(r'\s+[\w-]+=".*?"', '', text)
    text = re.sub(r'\s*(?:\d+\.|[-*+])\s+', ' ', text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"http?://\S+|www\.\S+|https?://\S+|pic\.twitter\.com/\S+", '', text)
    text = remove_hyphens(text)
    text = remove_currency_symbols(text)
    text = remove_bullet_points(text)

    CONTRACTION_MAP = {
        r"\bwon\'t\b": "will not",
        r"\bcan\'t\b": "cannot",
        r"\bshan\'t\b": "shall not",
        r"\bain\'t\b": "is not",
        
        r"\bcan t\b": "cannot",
        r"\bdon t\b": "do not",
        r"\bit s\b": "it is",
        r"\bi m\b": "i am",

        r"n\'t\b": " not",
        r"\'re\b": " are",
        r"\'s\b": " is",     
        r"\'d\b": " would",   
        r"\'ll\b": " will",
        r"\'t\b": " not",
        r"\'ve\b": " have",
        r"\'m\b": " am",
    }

    SLANG_MAP = {
        r"\bgonna\b": "going to",
        r"\bgon na\b": "going to",
        r"\bwanna\b": "want to",
        r"\bwan na\b": "want to",
        r"\brn\b": "right now",
        r"\bidk\b": "i do not know",
        r"\bim\b": "i am",      
        r"\bu\b": "you",
        r"\bur\b": "your",
        r"\bthx\b": "thanks",
    }

    
    FULL_MAP = {**CONTRACTION_MAP, **SLANG_MAP}
    for pattern, replacement in FULL_MAP.items():
        text = re.sub(pattern, replacement, text)

    punc_list = [p for p in ['!\xad\xa0\xe2\x80\x9d\xe2\x80\x99\xe2\x80\xa2()*+-/:;=[]^_`{|}~@#,.?$%&"”“’‘\'…«»–']]
    punc_str = "".join(punc_list)
    safe_punc_pattern = rf'[ {re.escape(punc_str)} ]'
    text = re.sub(safe_punc_pattern, ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_points(text):
    pattern = r'<(p|li)[^>]*>(.*?)</\1>'
    
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    
    if matches:
        cleaned_points = []
        for tag, content in matches:
            clean_content = re.sub(r'<.*?>', ' ', content)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()

            cleaned_points.append(clean_content)
            
        return '. '.join(cleaned_points)
    else:
        clean_content = re.sub(r'<.*?>', ' ', text)
        return re.sub(r'\s+', ' ', clean_content).strip()

def add_features(X):
    X_feat = X.copy()
    X_feat['full_text_raw'] = X_feat['company_profile'] + ' - ' + X_feat['description'] + ' - ' + X_feat['requirements'] + ' - ' + X_feat['benefits']
    X_feat['word_count'] = X_feat['full_text_raw'].apply(lambda x: len(word_tokenize(x)))
    X_feat['sentence_count'] = X_feat['full_text_raw'].apply(lambda x: len(sent_tokenize(x)))
    X_feat['total_emojis'] = X_feat['full_text_raw'].apply(lambda x: len(emoji_pattern.findall(x)))
    X_feat['sentiment_polarity'] = X_feat['full_text_raw'].apply(lambda x: TextBlob(x).sentiment.polarity)
    X_feat['sentiment_subjectivity'] = X_feat['full_text_raw'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    punc_count = lambda x: len([char for char in x if char in string.punctuation])
    X_feat['punc_ratio'] = X_feat['full_text_raw'].apply(lambda x: punc_count(x)) / X_feat['word_count'].replace(0, 1)
    X_feat['lexical_diversity'] = X_feat['full_text_raw'].apply(lambda x: len(set(word_tokenize(x)))) / X_feat['word_count'].replace(0, 1)
    X_feat['number_ratio'] = X_feat['full_text_raw'].apply(lambda x: len(re.findall(r'\d+', x))) / X_feat['word_count'].replace(0, 1)
    new_cols = ['word_count', 'sentence_count', 'total_emojis', 'sentiment_polarity', 'sentiment_subjectivity', 'punc_ratio', 'lexical_diversity', 'number_ratio']
    return X_feat[new_cols]
   
def preprocess(df):
    binary_cols = ['telecommuting', 'has_company_logo', 'has_questions', 'in_balanced_dataset']
    mapping = {'f': 0, 't': 1}

    for col in binary_cols:
        df[col] = df[col].map(mapping).astype('int')

    cols_to_drop = ['title', 'department', 'industry']
    df.drop(columns=cols_to_drop, inplace=True)

    df['country'] = df['location'].apply(lambda x: x.split(',')[0] if pd.notnull(x) else x)
    df.drop(columns=['location'], inplace=True)
    available_country = ['US', 'GB', 'GR', 'CA', 'DE', 'NZ', 'IN', 'AU', 'PH', 'NL', 'BE', 'IE',
       'SG', 'HK', 'PL', 'IL', 'EE', 'FR', 'ES', 'AE', 'EG', 'SE', 'RO', 'DK',
       'ZA', 'BR', 'IT', 'FI', 'PK', 'LT', 'MY', 'QA', 'JP', 'RU', 'PT', 'MX',
       'TR', 'BG', 'CH', 'SA', 'CN', 'HU', 'AT', 'MU', 'MT', 'UA', 'ID', 'CY',
       'NG', 'TH', 'KR', 'IQ']
    
    df['country'] = df['country'].apply(lambda x: x if x in available_country else 'other')
    df['has_salary'] = df['salary_range'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df.drop(columns=['salary_range'], inplace=True)

    text_col = ['company_profile', 'description', 'requirements', 'benefits']
    for col in text_col:
        df[f'{col}_cleaned'] = df[col].apply(cleaned_text)
        df[f'{col}_cleaned'] = df[f'{col}_cleaned'].apply(extract_points)
    
    df['full_text'] = df['company_profile_cleaned'] + ' - ' + df['description_cleaned'] + ' - ' + df['requirements_cleaned'] + ' - ' + df['benefits_cleaned']

    MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"  
    embedder = SentenceTransformer(MODEL_NAME)
    texts = df['full_text'].tolist()
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    cat_cols = ['employment_type',
                'required_experience',
                'required_education',
                'function',
                'country',
                'has_salary']
    
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)

    cat_features = encoder.transform(df[cat_cols])
    additional_features = add_features(df)

    X_final = np.hstack([embeddings, cat_features, additional_features.values])

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    X_final = scaler.transform(X_final)
    return X_final

    
    






