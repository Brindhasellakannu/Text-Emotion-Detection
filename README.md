# Text Emotion Detection Using Machine Learning

## Project Overview
This project focuses on detecting human emotions from text using Natural Language Processing (NLP) and Machine Learning techniques.

The system analyzes textual input and classifies it into different emotion categories.

## Technologies Used
- Python
- Natural Language Processing (NLP)
- Pandas
- Scikit-learn
- Joblib
- TF-IDF
- Bag of Words (BoW)

## Machine Learning Models
The project explores multiple classification algorithms:
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Classifier (Linear SVC)

## Project Workflow
1. Dataset collection and preprocessing
2. Text cleaning and stemming
3. Feature extraction using BoW and TF-IDF
4. Model training and evaluation
5. Emotion prediction from text input

## Model Performance
- Reported accuracy: 88%

## Project Structure
- `train_model.py` – Model training
- `evaluate_model.py` – Model evaluation
- `predict_emotion.py` – Emotion prediction
- `emotion_model.pkl` – Saved machine learning model
- `evaluation_report.txt` – Evaluation results
- `requirements.txt` – Required Python libraries

  ## Dataset

The project uses a dataset divided into three subsets:

- Training
- Testing
- Validation

The dataset files are not included in this repository. The trained model and project source code are provided for reference.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python predict_emotion.py
```

## Author
Brindha

B.Tech – Artificial Intelligence and Data Science
