import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

def main(infile: str, outdir: str, test_size: float = 0.2, random_state: int = 42):
    infile = Path(infile)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Loading data from: {infile}")
    df = pd.read_parquet(infile)
    # simple stratified split
    X = df['comment_clean'].fillna("").astype(str)
    y = df['target'].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    print(f"[INFO] Train rows: {len(X_train)}  Val rows: {len(X_val)}")

    # Vectorizer
    tf = TfidfVectorizer(max_features=100000, ngram_range=(1,2), stop_words='english')
    X_train_t = tf.fit_transform(X_train)
    X_val_t = tf.transform(X_val)

    # Classifier (balanced)
    clf = LogisticRegression(solver='saga', max_iter=2000, class_weight='balanced', n_jobs=-1)
    print("[INFO] Training LogisticRegression...")
    clf.fit(X_train_t, y_train)

     # Predict
    y_val_pred = clf.predict(X_val_t)
    y_val_proba = clf.predict_proba(X_val_t)[:,1]

    # Metrics
    print("[RESULT] Validation classification report:")
    print(classification_report(y_val, y_val_pred, digits=4))
    print("F1 (binary):", f1_score(y_val, y_val_pred))
    print("Precision:", precision_score(y_val, y_val_pred))
    print("Recall:", recall_score(y_val, y_val_pred))

    # Save artifacts
    joblib.dump(tf, outdir / "tfidf.joblib")
    joblib.dump(clf, outdir / "baseline_lr.joblib")
    # save validation predictions for error analysis
    val_df = pd.DataFrame({
        'comment_raw': X_val.values,
        'label': y_val.values,
        'pred': y_val_pred,
        'proba': y_val_proba
    })
    val_df.to_csv(outdir / "val_preds.csv", index=False)
    print(f"[INFO] Saved models and val_preds to {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, default="data/processed/train_clean.parquet", help="Processed parquet")
    parser.add_argument("--outdir", type=str, default="models", help="Output directory for models and preds")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    main(args.infile, args.outdir, test_size=args.test_size, random_state=args.random_state)