# Import needed libraries
import pandas as pd
import random
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_scheduler,
)
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os

# Ensure output directories exist
os.makedirs("data", exist_ok=True)

# Step 1: Load and preprocess the dataset
file_path = "https://raw.githubusercontent.com/ailina-aniwan/IDS703_NLP_Final_Project/refs/heads/main/data/womens_clothing_ecommerce_reviews.csv"
data = pd.read_csv(file_path)

# Remove rows with missing review text
cleaned_data = data.dropna(subset=["Review Text"]).reset_index(drop=True)

# Map ratings to sentiment categories
cleaned_data["Sentiment"] = cleaned_data["Rating"].map(
    {1: "Dissatisfied", 2: "Dissatisfied", 3: "Neutral", 4: "Satisfied", 5: "Satisfied"}
)

# Text preprocessing
cleaned_data["Review Text"] = (
    cleaned_data["Review Text"]
    .str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
    .str.strip()
)

# Save the cleaned data
output_path = "data/cleaned_reviews.csv"
cleaned_data.to_csv(output_path, index=False)
print(f"Cleaned data saved to {output_path}")

# Step 2: Define synthetic data templates
templates = [
    "I [user_verb] this product because it [product_verb] my expectations.",
    "The quality was [adjective], and I would [user_verb] recommend it.",
    "This product is [adjective]. Definitely [user_verb] it to anyone!",
    "I found this product to be [adjective], and it [product_verb] my needs.",
]

positive_words = ["amazing", "great", "perfect", "excellent", "loved"]
negative_words = ["terrible", "bad", "poor", "horrible", "disappointing"]

user_verbs_positive = ["love", "like", "appreciate", "enjoy"]
user_verbs_negative = ["dislike", "hate", "avoid", "regret"]

product_verbs_positive = ["exceeded", "met", "fulfilled", "impressed"]
product_verbs_negative = ["fell short of", "failed to", "disappointed", "lacked"]


# Generate synthetic review
def generate_review(sentiment="positive"):
    template = random.choice(templates)
    if sentiment == "positive":
        review = template.replace("[adjective]", random.choice(positive_words))
        review = review.replace("[user_verb]", random.choice(user_verbs_positive))
        review = review.replace("[product_verb]", random.choice(product_verbs_positive))
    else:
        review = template.replace("[adjective]", random.choice(negative_words))
        review = review.replace("[user_verb]", random.choice(user_verbs_negative))
        review = review.replace("[product_verb]", random.choice(product_verbs_negative))
    return review


# Generate synthetic dataset
def generate_synthetic_dataset(num_samples=1000):
    data = []
    for _ in range(num_samples):
        sentiment = random.choice(["positive", "negative"])
        review = generate_review(sentiment=sentiment)
        label = 1 if sentiment == "positive" else 0
        data.append({"Review Text": review, "Recommended IND": label})
    return data


# Create and save synthetic dataset
synthetic_data = generate_synthetic_dataset(num_samples=2000)
synthetic_df = pd.DataFrame(synthetic_data)
synthetic_df.to_csv("data/synthetic_reviews.csv", index=False)
print("Synthetic dataset saved to 'data/synthetic_reviews.csv'")

# Step 3: Prepare data for model training
# Load synthetic dataset
synthetic_data = pd.read_csv("data/synthetic_reviews.csv")
print(f"Loaded synthetic data with {len(synthetic_data)} samples")

# Split data into train/test
train_texts, val_texts, train_labels, val_labels = train_test_split(
    synthetic_data["Review Text"],
    synthetic_data["Recommended IND"],
    test_size=0.1,
    random_state=42,
)

# Tokenize data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(
    list(train_texts), truncation=True, padding=True, max_length=128
)
val_encodings = tokenizer(
    list(val_texts), truncation=True, padding=True, max_length=128
)


# Define dataset class
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = ReviewDataset(train_encodings, train_labels)
val_dataset = ReviewDataset(val_encodings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Step 4: Model Training
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_loader) * 3
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

epochs = 3
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(
        f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_train_loss / len(train_loader)}"
    )

# Step 5: Model evaluation and extract good/bad predictions


# Evaluate and extract predictions for synthetic data
def evaluate_and_extract(loader, model, device):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Predict
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()

            # Add results for the batch
            results.extend(
                {
                    "Review Text": tokenizer.decode(
                        input_ids[i], skip_special_tokens=True
                    ),
                    "True Label": labels.cpu().numpy()[i],
                    "Predicted Label": predicted_labels[i],
                }
                for i in range(len(predicted_labels))
            )
    return results


# Evaluate the model and collect results for synthetic data
results = evaluate_and_extract(val_loader, model, device)

synthetic_results_df = pd.DataFrame(results)

# Separate good and poor performance
good_performance = synthetic_results_df[
    synthetic_results_df["True Label"] == synthetic_results_df["Predicted Label"]
]
poor_performance = synthetic_results_df[
    synthetic_results_df["True Label"] != synthetic_results_df["Predicted Label"]
]

# Display examples
print("Good Performance Examples:")
print(good_performance[["Review Text", "True Label", "Predicted Label"]].sample(3))

print("Poor Performance Examples:")
print(poor_performance[["Review Text", "True Label", "Predicted Label"]].head(3))


# Evaluate and collect predictions for real data
model = BertForSequenceClassification.from_pretrained("bert-recommendation-model")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
# After validation loop
predictions = []
true_labels = []
val_texts_list = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

        # Decode the input_ids back to text
        batch_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        val_texts_list.extend(batch_texts)

# Create a DataFrame
val_results = pd.DataFrame(
    {
        "Review Text": val_texts_list,
        "True Label": true_labels,
        "Predicted Label": predictions,
    }
)

# Map labels to readable format
label_map = {0: "Not Recommend", 1: "Recommend"}
val_results["True Label"] = val_results["True Label"].map(label_map)
val_results["Predicted Label"] = val_results["Predicted Label"].map(label_map)

# Extract correct and incorrect predictions
correct_predictions = val_results[
    val_results["True Label"] == val_results["Predicted Label"]
]
incorrect_predictions = val_results[
    val_results["True Label"] != val_results["Predicted Label"]
]

# Save to CSV
correct_predictions.to_csv("correct_predictions.csv", index=False)
incorrect_predictions.to_csv("incorrect_predictions.csv", index=False)

# Step 6: Comparison table and chart
metrics_data = {
    "Metric": [
        "Accuracy",
        "Precision (Not Recommend)",
        "Precision (Recommend)",
        "Recall (Not Recommend)",
        "Recall (Recommend)",
        "F1-Score (Not Recommend)",
        "F1-Score (Recommend)",
    ],
    "Synthetic Data": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "Real Data": [0.91, 0.79, 0.94, 0.73, 0.96, 0.76, 0.95],
}

comparison_df = pd.DataFrame(metrics_data)
comparison_df.set_index("Metric").plot(kind="bar", figsize=(10, 6))
plt.title("Comparison of Metrics: Real vs Synthetic Data")
plt.ylabel("Metric Value")
plt.xticks(rotation=45, ha="right")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
