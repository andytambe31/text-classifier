from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = [
    "I love this",
    "This is great",
    "I hate this",
    "This is bad"
]

labels = [
    "positive",
    "positive",
    "negative",
    "negative"
]

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

new_texts = [
    "I love this",
    "This is bad",
    "This is great"
]

X_new = vectorizer.transform(new_texts)
predictions = model.predict(X_new)

print("\nPredictions:")
for text, prediction in zip(new_texts, predictions):
    print(text, "->", prediction)

print(model.classes_)
print(model.predict_proba(X_new))