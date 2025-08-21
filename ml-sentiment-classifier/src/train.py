import re
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Загрузка данных
reviews = load_files("data/aclImdb/train/", categories=["pos", "neg"])
X, y = reviews.data, reviews.target

# Предобработка текста
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


X_clean = [preprocess(doc.decode("utf-8")) for doc in X]

# Векторизация
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X_clean)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Обучение модели
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Оценка качества
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#  Интерпретация признаков
feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]

top_pos = np.argsort(coef)[-10:]
top_neg = np.argsort(coef)[:10]

print("Top positive words:", feature_names[top_pos])
print("Top negative words:", feature_names[top_neg])
