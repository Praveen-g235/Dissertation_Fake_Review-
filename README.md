
# Evaluating Machine Learning Models for Fake Review Detection

## Project Title

Evaluating Machine Learning Models for Fake Review Detection Using TF-IDF Representation, Textual Analysis, Explainable Machine Learning, and Prototype Development

# Author Name : Praveen Gokanakonda 
# Student id : c4045481 
## Project Overview

This project develops and evaluates machine learning models for classifying online reviews as computer generated or original reviews. The implementation uses natural language processing, TF-IDF feature representation, supervised machine learning, hyperparameter tuning, model evaluation, SHAP analysis, LIME analysis, and a prototype.

The target variable is label.

The two classes are:

CG: Computer Generated Review
OR: Original Review

## Dataset

The project uses the Dataset:

https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset

Initial dataset size:

40432 rows
4 columns

Columns:

category
rating
label
text_

Target variable:

label

Independent variables:

category
rating
text_

The dataset initially contained 12 duplicate records and no missing values.

After data preprocessing, the dataset contained:

40420 rows
4 original columns

Text preprocessing then added the Processed_Review column.

## Data Preprocessing

The following preprocessing steps are implemented in the notebook:

1. Missing values are checked and removed.
2. Duplicate records are identified and removed.
3. Review text is converted to lowercase.
4. URLs are removed.
5. HTML tags are removed.
6. Numbers are removed.
7. Punctuation and special characters are removed.
8. Extra white spaces are removed.
9. English stop words are removed.
10. WordNet lemmatization is applied.
11. The processed text is stored in the Processed_Review column.

After text preprocessing:

Number of processed reviews: 40420
Average number of words per processed review: 31.58
Missing values in processed reviews: 0

## Exploratory Data Analysis

The notebook performs exploratory analysis using:

- Class distribution analysis
- Review length analysis
- Word clouds for computer generated and original reviews
- Review length comparison by class
- Top 20 word frequency analysis for both classes

## TF-IDF Feature Representation

TF-IDF is used to convert processed review text into numerical features.

Configuration:

TfidfVectorizer(max_features=5000)

TF-IDF feature matrix:

40420 rows
5000 features

Training feature matrix:

32336 rows
5000 features

Testing feature matrix:

8084 rows
5000 features

The TF-IDF matrix contains 932266 non-zero values and has approximately 99.54 percent sparsity.

## Train Test Split

The dataset is divided using:

Test size: 20 percent
Random state: 42
Stratification: enabled

Training data:

32336 records

Testing data:

8084 records

Training class distribution:

OR: 16172
CG: 16164

Testing class distribution:

OR: 4043
CG: 4041

The classes are therefore approximately balanced in both the training and testing sets.

## Machine Learning Models

Four baseline machine learning models are implemented:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Random Forest
4. XGBoost

The models are evaluated using:

Accuracy
Precision
Recall
F1-score
ROC-AUC

## Baseline Model Results

The baseline results obtained from the notebook are:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8695 | 0.8749 | 0.8622 | 0.8685 | 0.9429 |
| Multinomial Naive Bayes | 0.8521 | 0.8362 | 0.8755 | 0.8554 | 0.9354 |
| Random Forest | 0.8431 | 0.8213 | 0.8770 | 0.8483 | 0.9261 |
|XGBoost	      | 0.8382	|0.8548	 |0.8146	|0.8343	| 0.9260|

The notebook also evaluates XGBoost as part of the model comparison and hyperparameter tuning workflow.

## Hyperparameter Tuning

GridSearchCV is used to tune the machine learning models.

Logistic Regression parameter search:

C: 0.1, 1, 10
solver: liblinear, lbfgs
Cross-validation: 5 folds

Multinomial Naive Bayes parameter search:

alpha: 0.1, 0.5, 1.0, 2.0
fit_prior: True, False
Cross-validation: 5 folds

Random Forest parameter search:

n_estimators: 100, 150
max_depth: 10, 20
min_samples_split: 2
Cross-validation: 3 folds

XGBoost parameter search:

n_estimators: 100, 150
max_depth: 3, 5
learning_rate: 0.1
Cross-validation: 3 folds

## Tuned Model Results

The final tuned model results are:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8712 | 0.8662 | 0.8780 | 0.8721 | 0.9476 |
| Multinomial Naive Bayes | 0.8521 | 0.8362 | 0.8755 | 0.8554 | 0.9354 |
| Random Forest | 0.8033 | 0.7984 | 0.8114 | 0.8049 | 0.8973 |
| XGBoost | 0.8110 | 0.8362 | 0.7733 | 0.8035 | 0.9061 |

## Best Model

The best tuned model is Logistic Regression.

The model was selected using the highest tuned F1-score.

Final performance:

Accuracy: 0.8712
Precision: 0.8662
Recall: 0.8780
F1-score: 0.8721
ROC-AUC: 0.9476

Best hyperparameters:

C: 10
solver: liblinear

Best cross-validation accuracy:

0.8735

The tuned Logistic Regression model also improved over the baseline Logistic Regression model in accuracy, F1-score, and ROC-AUC.

Baseline Logistic Regression:

Accuracy: 0.8695
F1-score: 0.8685
ROC-AUC: 0.9429

Tuned Logistic Regression:

Accuracy: 0.8712
F1-score: 0.8721
ROC-AUC: 0.9476

## Model Comparison: Default and Tuned Results

| Model | Default Accuracy | Tuned Accuracy | Default F1-score | Tuned F1-score | Default ROC-AUC | Tuned ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8695 | 0.8712 | 0.8685 | 0.8721 | 0.9429 | 0.9476 |
| Multinomial Naive Bayes | 0.8521 | 0.8521 | 0.8554 | 0.8554 | 0.9354 | 0.9354 |
| Random Forest | 0.8431 | 0.8033 | 0.8483 | 0.8049 | 0.9261 | 0.8973 |
| XGBoost | 0.8382 | 0.8110 | 0.8343 | 0.8035 | 0.9260 | 0.9061 |

## Explainable Machine Learning

The best tuned Logistic Regression model is analysed using SHAP and LIME.

SHAP analysis includes:

- SHAP beeswarm plot
- SHAP feature importance bar plot
- SHAP waterfall plot
- SHAP decision plot
- SHAP heatmap
- SHAP feature analysis

LIME is also used to provide a local explanation for an individual review.

For the selected LIME example, the model predicted:

Predicted class: CG
Probability of CG: 0.7879
Probability of OR: 0.2121

The LIME analysis displays the top contributing TF-IDF features for the selected prediction.

## Saved Files

The notebook saves the following files during execution:

best_logistic_model.pkl

This contains the best tuned Logistic Regression model.

prediction_results.csv

This contains the test-set reviews, actual labels, predicted labels, and whether each prediction was correct.

## Prototype

A  graphical user interface is implemented in the final notebook section.

The prototype loads prediction_results.csv and provides a simple interface for reviewing classification results.

The prototype displays:

- Review text
- Review category
- Review rating
- Actual review class
- Predicted review class
- Prediction status
- Navigation between reviews
- Direct review index selection

The class labels are displayed as:

CG: Computer Generated Review
OR: Original Review

## Software and Libraries

The implementation uses Python and the following libraries:

numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
nltk
wordcloud
shap
lime
joblib
tkinter

NLTK resources used:

stopwords
wordnet
omw-1.4

## Installation

Install the required Python packages using:

pip install numpy pandas matplotlib seaborn scikit-learn xgboost nltk wordcloud shap lime joblib


## Running the Project

1. Place the following files in the same working directory:

   Praveen Gokanakonda Final Dissertation Code.ipynb
   fake reviews dataset.csv

2. Open the notebook using Jupyter Notebook, JupyterLab, or another compatible environment.

3. Run the cells from top to bottom.

4. The notebook performs dataset inspection, preprocessing, exploratory analysis, TF-IDF feature extraction, model training, evaluation, hyperparameter tuning, model comparison, SHAP analysis, LIME analysis, and result generation.

5. The best model is saved as:

   best_logistic_model.pkl

6. Prediction results are saved as:

   prediction_results.csv

7. Run the final Tkinter section to open the classification results prototype.

## Evaluation Metrics

Accuracy measures the proportion of correctly classified reviews.

Precision measures the proportion of reviews predicted as a class that actually belong to that class.

Recall measures the proportion of actual reviews from a class that are correctly identified.

F1-score is the harmonic mean of precision and recall.

ROC-AUC measures the model's ability to distinguish between the two classes across classification thresholds.

## Project Output

The completed workflow produces:

- Cleaned review data
- Exploratory data analysis
- Processed review text
- TF-IDF feature matrix
- Baseline machine learning results
- Tuned machine learning results
- Model comparison tables
- Confusion matrices
- ROC-AUC analysis
- SHAP explanations
- LIME explanation
- Saved best model
- Prediction results CSV
- Tkinter classification prototype

## Reproducibility

The main train test split uses random_state=42 and stratified sampling. The machine learning models that specify random_state also use 42. Running the notebook from the beginning is recommended so that preprocessing, feature extraction, training, evaluation, saved files, and prototype outputs are generated consistently.

## Notes

The final selected model is the tuned Logistic Regression model with an F1-score of 0.8721 and ROC-AUC of 0.9476.

