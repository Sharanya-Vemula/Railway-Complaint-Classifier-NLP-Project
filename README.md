# 🚆 Public Transportation Complaint Classification Using NLP & Machine Learning

An NLP-based Machine Learning project that analyzes passenger complaints and automatically classifies them into meaningful categories. The goal is to simplify complaint management and help identify recurring service-related issues in Indian Railways.

## Project Overview

Passenger feedback plays an important role in improving transportation services. However, manually reviewing and categorizing a large volume of complaints can be time-consuming.

This project uses **Natural Language Processing (NLP)** and **Machine Learning** techniques to automatically classify passenger complaints into predefined categories based on their textual content.

The project follows a complete Machine Learning workflow, from data collection and preprocessing to model evaluation, model saving, and deployment through an interactive Gradio interface.

## Objectives

- Analyze passenger complaint data using NLP techniques.
- Clean and preprocess raw text data for Machine Learning.
- Classify complaints into relevant service-related categories.
- Implement and compare multiple Machine Learning algorithms.
- Evaluate model performance using classification metrics.
- Develop an interactive application for real-time complaint classification.

## Dataset Information

**Dataset:** [Indian Railways Complaint Tweets](https://huggingface.co/datasets/faizmubeen/indian_railways_complaint_tweets)

**Source:** Hugging Face

| Attribute | Details |
|---|---|
| Original records | 4,535 |
| Complaint records | 2,615 |
| Features | Tweet, Label, Type |
| Text column | Tweet |
| Complaint filter | Label = Complaint |
| Final records used | 1,376 |
| Final categories | 5 |

The original dataset contains both complaint and non-complaint tweets. Only complaint records were retained for the classification task.

**Note:** The five broad categories were created by mapping the original `Type` values using keyword-based rules. Records that could not be mapped were excluded from the final classification dataset.

## Complaint Categories

The project classifies complaints into five categories:

| Category | Description |
|---|---|
| Delays | Train delays, late arrivals, and punctuality-related issues |
| Cleanliness | Hygiene, sanitation, and cleanliness-related complaints |
| Staff Behaviour | Complaints related to staff conduct and behaviour |
| Ticketing | Ticket booking, reservation, cancellation, and refund issues |
| Safety | Passenger safety and security-related concerns |

## Technologies & Tools Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Natural Language Processing:** NLTK, Regular Expressions
- **Feature Extraction:** TF-IDF Vectorization
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Application Deployment:** Gradio
- **Dataset Platform:** Hugging Face

## Project Workflow

### 1. Business Understanding
Defined the problem of manually managing passenger complaints and identified the need for an automated classification system.

### 2. Data Loading & Exploration
- Loaded the dataset from Hugging Face.
- Examined dataset dimensions, columns, and data types.
- Checked missing values and duplicate records.
- Explored complaint distribution.

### 3. Data Preprocessing
- Filtered complaint records.
- Removed URLs, mentions, punctuation, and unnecessary characters.
- Converted text into lowercase.
- Removed extra whitespace.
- Tokenized text and removed stopwords while preserving important negation words.

### 4. Feature Extraction
Applied **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert textual complaints into numerical features suitable for Machine Learning models.

### 5. Model Building
Implemented and compared three Machine Learning algorithms:

- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

### 6. Model Evaluation
Evaluated the models using:
- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- Macro F1-score
- Confusion Matrix
- Classification Report

### 7. Model Selection
Compared the performance of all three algorithms and selected Logistic Regression based on its highest Macro F1-score.

### 8. Model Saving & Deployment
- Saved the trained model and TF-IDF vectorizer using Joblib.
- Developed an interactive complaint classification interface using Gradio.
- Enabled users to enter a complaint and receive a predicted category.

## Model Performance

The following results were obtained from the latest model evaluation:

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Naive Bayes | 64.49% | 57.49% | 64.49% | 54.92% | 30.82% |
| Logistic Regression | **86.23%** | **85.59%** | **86.23%** | **85.61%** | **74.25%** |
| SVM | 84.42% | 83.62% | 84.42% | 83.42% | 71.40% |

**Best Performing Model: Logistic Regression**

Logistic Regression achieved the highest Macro F1-score of **74.25%** and an accuracy of **86.23%** among the three evaluated models.

## 📁 Project Structure

```text
Public-Transportation-Complaint-Classification/
│
├── Public-Transportation-Complaint-Classification-NLP-Project.ipynb
│
├── best_model.pkl
│
├── tfidf_vectorizer.pkl
│
 
```

*The structure above is a suggested repository organization. Add the files and folders that are actually present in your GitHub repository.*

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project Folder

```bash
cd Public-Transportation-Complaint-Classification
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Jupyter Notebook

```bash
jupyter notebook
```

Open the project notebook and execute the cells sequentially.

### Step 5: Launch the Gradio Application

Run the Gradio application code in your notebook or application script.

```python
demo.launch()
```

Open the local URL displayed in the output to interact with the application.

## Key Learnings

- Understanding the practical applications of NLP in real-world problem-solving.
- Performing text cleaning and preprocessing.
- Converting text data into numerical features using TF-IDF.
- Implementing and comparing classification algorithms.
- Evaluating imbalanced multi-class classification using Macro F1-score.
- Saving trained Machine Learning models using Joblib.
- Building interactive Machine Learning applications using Gradio.

## Future Enhancements

- Expand the classification system to include bus and metro passenger feedback.
- Explore advanced NLP techniques and transformer-based models.
- Improve category mapping through manual validation and better labeling.
- Integrate multilingual complaint classification.
- Develop a dashboard to visualize complaint trends and category distributions.
- Explore integration with a real-time complaint management system.

## Author

**Shannu Vemula**

Machine Learning | Natural Language Processing | Python | Data Science

## 🔖 Tags

`NLP` `MachineLearning` `Python` `TextClassification` `TFIDF` `ScikitLearn` `Gradio` `DataScience` `ArtificialIntelligence`
