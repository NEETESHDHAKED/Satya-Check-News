import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
fake['label'] = 0 # 0 = Fake
true['label'] = 1 # 1 = Real

df = pd.concat([fake, true])
df = df.sample(frac=1).reset_index(drop=True)
print("Training Start...")
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(df['text'])
y = df['label']
model = LogisticRegression()
model.fit(X, y),
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("Training Complete. Model and Vectorizer saved as 'fake_news_model.pkl' and 'vectorizer.pkl' respectively.")
         