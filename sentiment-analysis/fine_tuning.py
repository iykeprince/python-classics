
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from datasets import Dataset, load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('slang_reviews.csv')

print(f'\n✅ Dataset loaded!')
print(f"    Total examples: {len(df)}")
print(f"    Columns: {list(df.columns)}")

# check balance
label_counts= df['label'].value_counts()
print(f"\n Dataset balance:")
print(f"    Negative(0): {label_counts[0]} {label_counts[0]/len(df)*100:.1f}%")
print(f"    Positive(1): {label_counts[1]} {label_counts[1]/len(df)*100:.1f}%")

# show first few examples
print(f"\n First 5 examples")
print(df.head())

print("\n" + "="*70)
print("PREPARING DATASET FOR TRAINING")
print("="*70)

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split into training(80%) and test(20%)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

print("\n✅ Dataset split:")
print(f"    Training examples: {len(dataset['train'])}")
print(f"    Testing examples: {len(dataset['test'])}")

# Verify the splits
train_labels = pd.Series(dataset['train']['label'])
test_labels = pd.Series(dataset['test']['label'])

print('\n📊 Training set balance')
print(f"     Negative: {(train_labels == 0).sum()}")
print(f"    Positive: {(train_labels == 1).sum()}")

print('\n📊 Test set balance')
print(f"    Negative: {(test_labels == 0).sum()}")
print(f"    Positive: {(test_labels == 1).sum()}")

print("\n"+"="*70)

print("\n"+"="*70)
print("LOADING PRE-TRAINED MODEL")
print("\n"+"="*70)

# Model name
model_name="distilbert-base-uncased-finetuned-sst-2-english"

print(f"\n🏠 loading model: {model_name}")

# Load tokenizer
tokenizer=AutoTokenizer.from_pretrained(model_name)
print("✅ Tokenizer loaded")

#load model
model= AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
print(" ✅  Model details:")
print(f"    Number of parameters: {model.num_parameters():,}")
print(f"    Number of labels: 2 (Negative, Positive)")

print("\n" + "="*70)

def tokenize_function(examples):
    """
    Converts text to tokens (numbers) that the model understands
    """
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )


print("\n🏠 Tokenizing training set...")
tokenize_train=dataset['train'].map(tokenize_function, batched=True)

print("\n😊 Tokenizing test set")
tokenize_test=dataset['test'].map(tokenize_function, batched=True)

print("\n ✅ Tokenization complete...")

# show examples
print(f" Example tokenization")
print(f"    Original text: {dataset['train'][0]['text']}")
print(f"    label: {dataset['train'][0]['label']}")
print(f"    Tokenized (first 20 tokens): {tokenize_train[0]['input_ids'][:20]}")
print(f"    Total tokens: {len(tokenize_train[0]['input_ids'])}")


print("\n" + "="*70)

print("\n" + "="*70)
print("CONFIGURING TRAINING PARAMETERS")
print("\n" + "="*70)

training_args=TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
    seed=42
)

print("\n✅ Training configuration:")
print(f"    Epoch: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"    Learning rate: {training_args.learning_rate}")
print(f"    Evaluation: After each epoch")

print("\n📊 Training will involve:")
train_steps=len(tokenize_train) // training_args.per_device_train_batch_size * training_args.num_train_epochs
print(f"    Approximately {train_steps} training steps")
print(f"    {len(tokenize_train) // training_args.per_device_train_batch_size} steps per epoch")
print("\n" + "="*70)

# Define evaluation metric
def compute_metrics(eval_pred):
    """
    Calculate accuracy during training
    """

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy':accuracy_score(labels, predictions)}


# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenize_train,
    eval_dataset=tokenize_test,
    compute_metrics=compute_metrics
)

print("\n 🚀 Beginning fine-tuning...")
print("     This will take 2 - 5 minutes depending on GPU allocation")
print("     You'll see progress update below:\n")

# Start training
trainer.train()

print("\n" + "="*70)
print("✅ FINE-TUNING COMPLETE!")
print("\n" + "="*70)


print("\n" + "="*70)
print("SAVING FINE-TUNED MODEL")
print("\n" + "="*70)

# Save model and tokeenizer
output_dir="./fine_tuned_sentiment_model"

print(f"\n Saving model to: {output_dir}")

print(" ✅ Model saved!")
print(" ✅ Tokenizer saved!")

import os

for file in os.listdir(output_dir):
    file_size = os.path.getsize(os.path.join(output_dir, file)) / (1024*1024) # Size in MB
    print(f"    (file) ({file_size:.2f} MB)")

# Create a zip file
print(f"\n🗜️ Creating zip file...")
!zip -r fine_tuned_sentiment_model.zip {output_dir}
print("   ✅ Zip file created!")


# # Download the zip file
# print(f"\n📥 Downloading model...")
# from google.colab import files
# files.download('fine_tuned_sentiment_model.zip')

# print("   ✅ Download started!")
# print("\n💡 The zip file should download to your computer's Downloads folder.")
# print("   You can extract it and use it later by loading it with:")
# print("   model = AutoModelForSequenceClassification.from_pretrained('path/to/extracted/folder')")
# print("   tokenizer = AutoTokenizer.from_pretrained('path/to/extracted/folder')")

# print("\n" + "="*70)