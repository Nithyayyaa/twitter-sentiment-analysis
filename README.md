# Twitter Sentiment Analysis — Classifier Comparison Study

An NLP sentiment classifier benchmarking four ML algorithms on 31,962 
labeled tweets, achieving **95.60% accuracy with Random Forest + TF-IDF**. 
Deployed as a live Flask web application for real-time tweet sentiment 
prediction. Built as a minor project (IS6C06) at NIE Mysuru, under 
Visvesvaraya Technological University.

## Problem

Social media produces vast amounts of unstructured opinion data. Manually 
classifying tweet sentiment at scale is impossible, and automated tools 
enable brand monitoring, content moderation, and public-opinion analysis. 
This project systematically compares four classical ML approaches to 
identify the best-performing classifier for tweet sentiment.

## Approach

- **Dataset:** 31,962 labeled tweets from Kaggle (binary positive/negative)
- **Preprocessing:** Text normalization (regex), lemmatization, 
  stopword removal, hashtag/special-character stripping
- **Feature extraction:** TF-IDF vectorization (chosen over CountVectorizer 
  for its ability to weight word importance, not just frequency)
- **Classifiers compared:** Logistic Regression, Support Vector Machine, 
  K-Nearest Neighbors, Random Forest
- **Evaluation:** Accuracy, precision, recall, F1-score, and confusion 
  matrix (balancing True Positive Rate against True Negative Rate)
- **Deployment:** Flask web app with real-time Twitter data via Tweepy API

## Results

| Classifier | Accuracy | TPR | TNR |
|------------|----------|------|------|
| **Random Forest** | **95.60%** | 0.9962 | **0.4421** |
| SVM | 95.38% | 0.9988 | 0.3642 |
| Logistic Regression | 94.50% | 0.9985 | 0.2378 |
| KNN | 93.69% | 0.9999 | 0.1313 |

**Random Forest was selected** as the final model. Although SVM had 
slightly higher TPR, Random Forest achieved the highest accuracy AND 
the best True Negative Rate (0.4421), giving it the most balanced 
performance on this imbalanced dataset.

## Tech Stack

**ML:** Python · scikit-learn · NLTK · TF-IDF · pandas · NumPy  
**Visualization:** matplotlib · WordCloud  
**Web App:** Flask · HTML/CSS · Tweepy (Twitter API) · Google Cloud NLP

## Repository Structure

├── notebooks/
│   ├── Schmaltz-Surveyor.ipynb       Main classifier comparison
│   ├── data_exploration.ipynb         EDA, word clouds, hashtag analysis
│   └── final_model.ipynb              Final Random Forest training
├── webapp/                            Flask app for live prediction
├── data/sample_tweets.csv             Sample from training dataset
└── report/
├── project_report.pdf             Full methodology and results
└── presentation.pdf               Project presentation

## Running Locally

```bash
pip install -r requirements.txt
jupyter notebook notebooks/final_model.ipynb    # train classifier
cd webapp && python app.py                      # launch web app
```

Trained model pickle files are not included due to size; regenerate by 
running the training notebook.

## My Contributions

This was a 4-person group project (IS6C06 Minor Project, NIE Mysuru, 
2021–22). My specific contributions:

- **TF-IDF vectorization pipeline** — feature extraction from cleaned tweets
- **Classifier training and tuning** — implementation and hyperparameter 
  tuning for SVM, Logistic Regression, and contribution to Random Forest
- **Model evaluation and selection** — accuracy benchmarking, confusion 
  matrix analysis, TPR/TNR balance comparison leading to Random Forest 
  selection

**Team:** Nithyashree Arunachalam · Pradyoth P · Shashank BU · Tejasvini SJ  
**Guide:** Mrs. Nandini BM, Assistant Professor, Dept. of ISE

## Future Work (from original report)

- Extend to regional (non-English) languages
- Integrate as a content-moderation plugin for live platforms
- Compare with modern transformer-based approaches (BERT, RoBERTa)

## Author

**Nithyashree Arunachalam**  
Master's student in Data & Knowledge Engineering, OvGU Magdeburg

## My Contributions

This was a 4-person group project (IS6C06 Minor Project, NIE Mysuru, 2021–22). 
My specific contributions:

- **Text preprocessing pipeline** — regex-based cleaning, Porter stemming, 
  NLTK stopword removal, and TF-IDF vectorization
- **Classifier training** — implementation and evaluation of SVM (RBF kernel), 
  Logistic Regression, KNN, and Random Forest using scikit-learn
- **Model comparison and selection** — accuracy benchmarking and 
  TPR/TNR confusion matrix analysis; Random Forest selected based on 
  best balanced performance
- **Documentation** — methodology write-up and results comparison

**Team:** Nithyashree Arunachalam · Pradyoth P · Shashank BU · Tejasvini SJ  
**Guide:** Mrs. Nandini BM, Assistant Professor, Dept. of ISE
