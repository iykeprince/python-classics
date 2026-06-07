import os
# Silences the unauthenticated token warning if you don't want to log in
os.environ["HF_TOKEN"] = "" 

from transformers import pipeline
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load a pre-trained sentiment analysis model
print('Loading model...')
classifier = pipeline("sentiment-analysis",
                      model="distilbert-base-uncased-finetuned-sst-2-english")

# Load IMDB dataset (we'll use a small subset for speed)
print('Loading dataset...')
dataset = load_dataset("stanfordnlp/imdb", split="test[:1000]")
dataset = dataset.shuffle(seed=42) # First 1000 test examples

# Function to predict on batch
def predict_sentiment(texts, batch_size=32):
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        results = classifier(batch, truncation=True, max_length=512)
        
        # FIX: Changed .extends() to .extend()
        predictions.extend(results) 
    return predictions

# Get Predictions
# print("Make predictions on a 1000 reviews")
texts = dataset['text']
predictions = predict_sentiment(texts)

# # Convert predictions to binary (POSITIVE=1, NEGATIVE=0)
pred_labels=[1 if p['label'] == 'POSITIVE' else 0 for p in predictions]
true_labels=dataset['label']

# # Evaluate our model
# accuracy = accuracy_score(true_labels, pred_labels)
# print(f'Accuracy score: {accuracy:.4f}')

# print(classification_report(true_labels, pred_labels, 
#                             target_names=['Negative', 'Positive']))

# # Show some examples
# print("\n--- Example Predictions ---")
# for i in range(5):
#     print(f"\nReview: {texts[i][:200]}...")
#     print(f"True: {'Positive' if true_labels[i] == 1 else 'Negative'}")
#     print(f"Predicted: {predictions[i]['label']} (confidence: {predictions[i]['score']:.3f})")

# Calculate confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Create visualization
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Negative','Positive'],
    yticklabels=['Negative','Positive'],
    cbar_kws={'label': 'Number of Reviews'},
    annot_kws={'size': 16, 'weight':'bold'}
)

plt.xlabel('Predicted Sentiment', fontsize=12, fontweight='bold')
plt.ylabel('Actual Sentiment', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix - Sentiment Analysis Results')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print('Confusion matrix saved as "confusion_matrix.png"')