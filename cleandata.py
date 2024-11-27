import pandas as pd

# Load the dataset
file_path = "/Users/liuliangcheng/Desktop/Duke/IDS_NLP/final/Womens Clothing E-Commerce Reviews.csv.zip"
data = pd.read_csv(file_path)

# Step 1: Remove rows with missing Review Text
cleaned_data = data.dropna(subset=["Review Text"]).reset_index(drop=True)

# Step 2: Map Ratings to Sentiment Categories
# (1-2: Dissatisfied, 3: Neutral, 4-5: Satisfied)
cleaned_data["Sentiment"] = cleaned_data["Rating"].map(
    {1: "Dissatisfied", 2: "Dissatisfied", 3: "Neutral", 4: "Satisfied", 5: "Satisfied"}
)

# Step 3: Text Preprocessing (optional for BERT)
# Remove special characters and strip extra spaces
cleaned_data["Review Text"] = (
    cleaned_data["Review Text"]
    .str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
    .str.strip()
)

# Step 4: Save the cleaned data
output_path = "/Users/liuliangcheng/Desktop/Duke/IDS_NLP/final/cleaned_reviews.csv"
cleaned_data.to_csv(output_path, index=False)

print(f"Cleaned data saved to {output_path}")


# The cleaned dataset appears to be in excellent condition for further modeling:

##Missing Values:

# No missing values in the Review Text column. No missing values in the Sentiment column. Total Entries:

# The dataset contains 22,641 entries, which is sufficient for fine-tuning a BERT model. Sentiment Distribution:

# Satisfied: 17,448 entries Neutral: 2,823 entries Dissatisfied: 2,370 entries While there is some imbalance favoring the "Satisfied" category, this can be managed during model training with techniques like class weighting or oversampling.
