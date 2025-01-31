import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelEncoder

# 1. Read the datasets
train_data = pd.read_csv('train_data.csv')  # Training set
val_data = pd.read_csv('val_data.csv')      # Validation set
test_data = pd.read_csv('test_data.csv')    # Test set

# Separate features and labels for train and validation
X_train, y_train = train_data['text'], train_data['label']
X_val, y_val = val_data['text'], val_data['label']

# Test: check if labels exist in the test data
if 'label' in test_data.columns:
    has_labels = True
    X_test, y_test = test_data['text'], test_data['label']
else:
    has_labels = False
    X_test = test_data['text']

# 2. Text preprocessing for Word2Vec
# Function to preprocess text into lists of words
def preprocess_text(text):
    return text.lower().split()

# Apply the function to all datasets
X_train_tokens = X_train.apply(preprocess_text)
X_val_tokens = X_val.apply(preprocess_text)
X_test_tokens = X_test.apply(preprocess_text)

# 3. Train the Word2Vec model
word2vec_model = Word2Vec(sentences=X_train_tokens, vector_size=100, window=5, min_count=1, workers=4, sg=1, seed=42)

# 4. Convert sentences into vectors by averaging word vectors
def vectorize_text(text_tokens, model):
    vectors = [model.wv[word] for word in text_tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Convert all datasets into vectors
X_train_vectors = np.array([vectorize_text(tokens, word2vec_model) for tokens in X_train_tokens])
X_val_vectors = np.array([vectorize_text(tokens, word2vec_model) for tokens in X_val_tokens])
X_test_vectors = np.array([vectorize_text(tokens, word2vec_model) for tokens in X_test_tokens])

# 5. Train the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_vectors, y_train)

# 6. Evaluate on the validation set
y_val_pred = log_reg.predict(X_val_vectors)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation set accuracy:", val_accuracy)
print("Classification report on validation set:")
classification_report_val = classification_report(y_val, y_val_pred, output_dict=True)

# 7. Predictions on the test set
y_test_pred = log_reg.predict(X_test_vectors)

if has_labels:
    # If the test set has labels, evaluate performance
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test set accuracy:", test_accuracy)
    print("Classification report on test set:")
    classification_report_test = classification_report(y_test, y_test_pred, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(cm)

    # 8. Visualizing the confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
    plt.title('Confusion Matrix')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.show()

    # 9. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, log_reg.predict_proba(X_test_vectors)[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    # 10. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, log_reg.predict_proba(X_test_vectors)[:, 1])

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    # 11. Visualizing Classification Report (Validation Set) as Heatmap
    def plot_classification_report(report, title='Classification Report'):
        report_df = pd.DataFrame(report).T
        plt.figure(figsize=(10, 8))
        sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap='Blues', fmt='.2f', cbar=False)
        plt.title(title)
        plt.show()

    # Plot classification report for validation and test sets
    plot_classification_report(classification_report_val, 'Validation Set Classification Report')
    plot_classification_report(classification_report_test, 'Test Set Classification Report')

else:
    # If the test set does not have labels, save predictions
    test_data['predicted_label'] = y_test_pred
    test_data.to_csv('test_predictions.csv', index=False)
    print("Predictions have been saved in 'test_predictions.csv'.")
