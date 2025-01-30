# Fake News Detection using NLP

## 📌 Overview
This project implements **Fake News Detection** using **Natural Language Processing (NLP)** and **Machine Learning techniques**. We explore various **traditional ML models, deep learning architectures, sentiment analysis, and transformer-based approaches** to classify news articles as **real or fake**.

## 🚀 Features
- Data preprocessing using **text normalization, tokenization, stopword removal, and lemmatization**.
- Multiple machine learning models including **Random Forest, SVM, and Logistic Regression**.
- Deep Learning approaches using **LSTMs, CNNs, and hybrid architectures**.
- **Sentiment Analysis** to explore how sentiment scores impact fake news classification.
- **Transformer-based models** including **BERT, RoBERTa, XLNet, and DistilBERT**.
- High-performing transformer models achieving **up to 99.5% accuracy**.

## 📂 Dataset
The dataset used is sourced from **Kaggle**, containing **6,335 articles** classified as **real** or **fake**.

| Feature | Description |
|---------|------------|
| `id` | Unique identifier for each article |
| `title` | The headline of the news article |
| `text` | The full body of the article |
| `label` | Fake (0) or Real (1) |

## 🏗 Methodologies Used
### 1️⃣ Traditional Machine Learning
- **Vectorization:** Word2Vec for embedding generation.
- **Models Tested:** Random Forest, SVM, and Logistic Regression.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score.
- **Best Model:** **SVM with 92% accuracy**.

### 2️⃣ Sentiment Analysis Approach
- Used **VADER and TextBlob** to analyze sentiment scores.
- Sentiment-based **XGBoost model achieved 72% accuracy**.
- Combined with embeddings, **accuracy improved to 93%**.

### 3️⃣ Deep Learning Approaches
- **LSTM-based RNN:** Achieved **87.5% accuracy**.
- **CNN:** Performed well with **84.6% accuracy**.
- **Hybrid CNN-RNN model:** Showed potential with **84.0% accuracy**.

### 4️⃣ Transformer Models
- **Fine-tuned BERT, RoBERTa, XLNet, and DistilBERT**.
- Best results:
  - **RoBERTa: 99.5% accuracy** 🎯
  - **BERT: 98.0% accuracy**
  - **XLNet: 98.8% accuracy**
  - **DistilBERT: 97.6% accuracy**

## 📊 Results & Performance
| Model | Accuracy | Precision | Recall |
|--------|----------|-----------|---------|
| Random Forest | 90% | 89% | 91% |
| SVM | 92% | 91% | 93% |
| LSTM | 87.5% | 86% | 88% |
| CNN | 84.6% | 83% | 85% |
| BERT | 98.0% | 97.8% | 98.1% |
| RoBERTa | 99.5% | 99.5% | 99.5% |
| XLNet | 98.8% | 98.7% | 98.9% |
| DistilBERT | 97.6% | 97.4% | 97.7% |

## 📌 Future Improvements
- Expand dataset to support **multiple languages**.
- Develop a **browser extension** for real-time fake news detection.
- Optimize models for **faster inference and deployment**.

## 💡 Authors
- **Andrei-Ionut Popa** - [andrei-ionut.popa@s.unibuc.ro](mailto:andrei-ionut.popa@s.unibuc.ro)
- **Valentin-Ion Semen** - [valentin-ion.semen@s.unibuc.ro](mailto:valentin-ion.semen@s.unibuc.ro)
- **Elias-Valeriu Stoica** - [elias-valeriu.stoica@s.unibuc.ro](mailto:elias-valeriu.stoica@s.unibuc.ro)
- **Sabin-Sebastian Toma** - [sabin-sebastian.toma@s.unibuc.ro](mailto:sabin-sebastian.toma@s.unibuc.ro)

## 🌟 Acknowledgements
Special thanks to the **University of Bucharest** for supporting this research.

---
⭐ **If you find this project useful, don't forget to give it a star on GitHub!** ⭐
