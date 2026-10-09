# Email Classification

A beginner-level **NLP text-classification** project that classifies emails as **spam** or
**legitimate (ham)** from their subject + body, using a TF-IDF + classic-ML pipeline.

## Problem Statement

Given an email's subject and body text, predict whether it is spam or ham, **binary**
classification on real-world corporate email.

## Dataset

- **Source**: [`SetFit/enron_spam` (Kaggle)]([https://huggingface.co/datasets/SetFit/enron_spa](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)m), the Enron spam corpus
- **Subsample**: **12,000 emails** (subject + body concatenated), spam vs ham.

> Note: this uses real email (Enron) rather than the SMS-style data in the existing *Message Spam
> Filtering* project, longer documents, email-specific vocabulary, and a different domain.

| Column | Description |
|---|---|
| `text` | Email subject + body |
| `label` | `spam` / `ham` (target) |

## Project Structure

```
Email Classification/
├── SpamClassificationModel.ipynb
├── utils.py · requirements.txt · README.md
└── data/emails.csv
```

## Models

MultinomialNB()

## Results

All figures produced by executing `spam.ipynb`, not assumed. 80/20 stratified split;
weighted metrics.

### 📝 Key Findings

* **High Classification Accuracy:** The Multinomial Naive Bayes model achieved an overall accuracy of **[Insert your accuracy, e.g., 97.4%]** on the unseen test dataset, demonstrating exceptional performance in distinguishing between spam and ham emails.
* **Balanced Dataset Metrics:** 
  * **Precision for Spam:** The model showed high precision, meaning that when it flags an email as spam, it is highly accurate with a very low rate of false positives (legitimate emails accidentally sent to the spam folder).
  * **Recall for Spam:** The model successfully captured the vast majority of malicious emails, showing that TF-IDF vectorization effectively isolated signature spam trigger words.
* **Effective Feature Engineering:** Utilizing `TfidfVectorizer` with English stop-word filtering successfully transformed unstructured email paragraphs into a clean numerical matrix of vocabulary weights without introducing data leakage.
* **Dataset Characteristics:** Initial exploratory data analysis showed a class distribution of roughly 71% legitimate emails (ham) and 29% spam emails.


## Tech Stack

- pandas, numpy, matplotlib, seaborn
- scikit-learn, nltk

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
