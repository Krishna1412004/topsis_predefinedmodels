# TOPSIS-Based Evaluation of Pre-Trained Models for Text Classification

## Overview
This project applies the *TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)* method to evaluate and rank pre-trained transformer models for *IMDB Sentiment Classification*. The evaluation considers multiple performance metrics and selects the best model based on a weighted decision-making approach.

## Dataset
- *IMDB Sentiment Analysis Dataset* (from Hugging Face datasets library)
- Contains *movie reviews* labeled as *positive (1)* or *negative (0)*
- The dataset is *tokenized* using the bert-base-uncased tokenizer.
- Training set size: *1000 samples*
- Test set size: *200 samples*

## Models Evaluated
The following *pre-trained transformer models* are evaluated:

| Model     | Hugging Face Model Name |
|-----------|------------------------|
| *BERT*       | bert-base-uncased   |
| *RoBERTa*    | roberta-base        |
| *DistilBERT* | distilbert-base-uncased |
| *ALBERT*     | albert-base-v2 |

## Metrics Used
The models are evaluated based on the following performance metrics:
- *Accuracy*
- *Precision*
- *Recall*
- *F1-score*
- *Inference Time (Placeholder)*
- *Model Size (Placeholder)*

## TOPSIS Method
TOPSIS is used to rank the models based on the above metrics. The criteria are weighted as follows:

| Metric        | Weight |
|--------------|--------|
| Accuracy     | 0.30   |
| Precision    | 0.20   |
| Recall       | 0.20   |
| F1-score     | 0.20   |
| Inference Time | 0.05   |
| Model Size   | 0.05   |

### Steps:
1. *Dataset Loading & Tokenization*
2. *Fine-Tuning & Evaluation of Pre-Trained Models*
3. *Normalization & Weighting of Evaluation Metrics*
4. *Computing Ideal Best and Worst Values*
5. *Calculating TOPSIS Score & Ranking Models*
6. *Visualizing Results with a Bar Chart*

## Installation & Usage
### Prerequisites
Ensure you have Python installed along with the following libraries:
bash
pip install numpy pandas matplotlib transformers datasets scikit-learn torch


### Running the Script
Execute the following command:
bash
python topsis_text_classification.py


### Expected Output
1. *Table of Model Rankings* (Example Output):
bash
       Model  TOPSIS Score  Rank
0      BERT       0.7823     1
1  RoBERTa       0.7651     2
2  ALBERT       0.7486     3
3  DistilBERT       0.7204     4

2. *Bar Chart Visualization of Model Rankings*

## Results Interpretation
- *Higher TOPSIS score* indicates *better overall performance*.
- *Accuracy & F1-score* have the highest weights, influencing rankings the most.
- *Inference time & model size* have a minor impact due to their lower weights.

## Contributions
- Developed by: *[Your Name]*
- Institution: *Thapar Institute of Engineering and Technology*
- Contact: *[Your Email]*

## License
This project is open-source and available under the *MIT License*.

---
Feel free to modify this README as needed. 🚀
