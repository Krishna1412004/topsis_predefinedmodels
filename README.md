# TOPSIS Evaluation of Pre-trained Models for Sentiment Classification

## Overview

In this project, we evaluated several pre-trained models for sentiment classification on the IMDB dataset using the **TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)** method. We utilized four different models from Hugging Face's `transformers` library: BERT, RoBERTa, DistilBERT, and ALBERT. The models were compared based on multiple performance metrics including accuracy, precision, recall, F1-score, inference time, and model size.

## Steps

### 1. Install Required Libraries

The following libraries are required for the project:

- `transformers`
- `datasets`
- `scikit-learn`
- `numpy`
- `matplotlib`

You can install them using the following command:

```bash
pip install transformers datasets scikit-learn numpy matplotlib
2. Load the IMDB Dataset
We load the IMDB dataset using the datasets library. The dataset contains movie reviews labeled as positive or negative.

python
Copy
Edit
from datasets import load_dataset

dataset = load_dataset("imdb")
3. Tokenization
We used the AutoTokenizer from the transformers library to tokenize the input text and convert it into the format required by the models.

python
Copy
Edit
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
4. Model Selection
We compared the following models:

BERT (bert-base-uncased)
RoBERTa (roberta-base)
DistilBERT (distilbert-base-uncased)
ALBERT (albert-base-v2)
Each model was used for training and evaluation on the IMDB dataset.

5. Model Training and Evaluation
Each model was trained and evaluated using the Trainer class from the transformers library. We evaluated the models on the test set and computed performance metrics such as accuracy, precision, recall, and F1-score.

python
Copy
Edit
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
6. Apply TOPSIS
We applied the TOPSIS method to rank the models based on their performance. The following metrics were considered:

Accuracy
Precision
Recall
F1-score
Inference Time
Model Size
The models were normalized, weighted, and evaluated using the TOPSIS score. The final ranking was based on these scores.

python
Copy
Edit
# Normalize the data, calculate ideal best and worst, and compute TOPSIS scores
norm_data = np.nan_to_num(data / np.sqrt((data**2).sum(axis=0)))
weighted_data = norm_data * weights
ideal_best = np.max(weighted_data, axis=0) * benefit_criteria + np.min(weighted_data, axis=0) * (1 - np.array(benefit_criteria))
ideal_worst = np.min(weighted_data, axis=0) * benefit_criteria + np.max(weighted_data, axis=0) * (1 - np.array(benefit_criteria))

distance_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
distance_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))

topsis_scores = np.nan_to_num(distance_worst / (distance_best + distance_worst))
rankings = np.argsort(topsis_scores)[::-1] + 1
7. Results
The models were ranked based on the TOPSIS scores, and a bar chart was generated to visualize the performance.

python
Copy
Edit
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.bar(results_df['Model'], results_df['TOPSIS Score'], color=['blue', 'green', 'red', 'purple'])
plt.xlabel("Models")
plt.ylabel("TOPSIS Score")
plt.title("TOPSIS Ranking of Text Classification Models")
plt.show()
8. Final Rankings
The final rankings of the models based on the TOPSIS evaluation are displayed in a table:

python
Copy
Edit
results_df = pd.DataFrame({
    'Model': list(models.keys()),
    'TOPSIS Score': topsis_scores,
    'Rank': rankings
}).sort_values(by='TOPSIS Score', ascending=False)

print(results_df)
Folder Structure
bash
Copy
Edit
.
├── X_train.csv                  # Training data
├── X_test.csv                   # Test data
├── submission.csv               # Predictions for submission
└── model_script.py              # Python script for preprocessing, training, and evaluation
Conclusion
This project demonstrated the application of TOPSIS for evaluating pre-trained models on a sentiment classification task. By using this method, we were able to rank multiple models based on various performance metrics, helping to identify the best performing model for the task.

go
Copy
Edit

### Key Formatting:

1. **Headings**: Use `##` for sections and `###` for subsections.
2. **Code Blocks**: Use triple backticks (```) for code snippets.
3. **Lists**: Use `-` for bullet points or `1.` for ordered lists.
4. **Inline Code**: Wrap function names or model names in single backticks (`).

This will create a structured and readable README for your project. You can copy and paste this directly into your project’s README file.
