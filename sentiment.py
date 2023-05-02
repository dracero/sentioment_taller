#Clasificación en positivo, negativo y neutro
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("finiteautomata/beto-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/beto-sentiment-analysis")

# Define a function to predict the sentiment of a given text
def predict_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    print(text)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=1)
    sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
    predicted_sentiment = sentiment_labels[predictions.item()]
    return predicted_sentiment

# Read the Excel file
df = pd.read_excel("sentimiento.xlsx")
df = df.dropna(subset=['review'])
df['review'] = df['review'].astype(str)

# Create a new column named "sentimiento" in the dataframe
df["sentimiento"] = ""

# Define the maximum number of threads to use
max_workers = 15

# Use multithreading to predict the sentiment of each text
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_index = {executor.submit(predict_sentiment, row["review"]): index for index, row in df.iterrows()}
    for future in concurrent.futures.as_completed(future_to_index):
        index = future_to_index[future]
        try:
            predicted_sentiment = future.result()
            df.loc[index, "sentimiento"] = predicted_sentiment
        except Exception as exc:
            print(f"Error en la fila {index}: {exc}")

# Save the modified dataframe back to the Excel file
df = df[['review','sentimiento']]
df.to_excel("data.xlsx")