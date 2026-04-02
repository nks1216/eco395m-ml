import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import RidgeClassifierCV

if __name__ == "__main__":

    df = pd.read_csv("data/US-Economic-News.csv",  encoding='latin-1').set_index("articleid")
    
    y = [0 if not label == "yes" else 1 for label in df["relevance"].values]
    X = df["text"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = make_pipeline(CountVectorizer(), TfidfTransformer(), RidgeClassifierCV())

    model.fit(X_train, y_train)

    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    print(classification_report(y_true=y_test, y_pred=y_hat_test))
