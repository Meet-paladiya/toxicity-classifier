import joblib
tf = joblib.load("models/tfidf.joblib")
clf = joblib.load("models/baseline_lr.joblib")
texts = ["you are an idiot", "great play, well done!", "i will kill you"]
X = tf.transform(texts)
probs = clf.predict_proba(X)[:,1]
for t,p in zip(texts,probs):
    print(f"{p:.3f}  {t}")