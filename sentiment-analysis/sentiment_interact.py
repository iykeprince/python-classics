from transformers import pipeline
# Load a pre-trained sentiment analysis model
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("Model loaded successfully")

def analyze_custom_text():
    print("\n" + "="*50)
    print("Custom Sentiment Analysis")
    print("="*50)
    print("Enter movie reviews to analyze (or 'quit' to exit)\n")
    
    while True:
        user_input = input("Enter review: ").strip()
        if user_input.lower() == 'quit':
            break
        
        if user_input:
            result = classifier(user_input)[0]
            print(f"Sentiment: {result['label']}")
            print(f"Confidence: {result['score']:.2%}\n")

# Add this at the end of your script
if __name__ == "__main__":
    # Run evaluation first
    # ... (previous code)
    
    # Then allow custom input
    analyze_custom_text()