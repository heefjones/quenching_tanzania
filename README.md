# Overview:
<br>
Given various features of water wells in Tanzania, predict each well's condition/functionality. The possible values for well functionality are as follows:
  - Functional (class 0)
  - Functional, Needs repairs (class 1)
  - Non-functional (class 2)
The end goal is to build a predictive model that properly identifies each of these 3 classes with maximum accuracy.

### Goals:
<br>
  - Identify the features that correlate most with water well functionality.
  - Create a Logistic Regression model that predicts a water well's condition with maximum accuracy.

# Data
Given 39 features of a water well
The [data](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/) consisted of close to 60,000 water wells in Tanzania, recorded by GeoData Consultants Ltd.

# Data Cleaning
I began with 

# Modeling
I began the modeling process with 3 simple models:
1. The baseline model (sklearn.DummyClassifier):
   - The baseline model simply predicted the most common class (functional), and predicted with 54% accuracy.
2. A simple logistic regression (sklearn.LogisticRegression):
   - This model used default parameters and predicted with 75% accuracy. Here is a confusion matrix displaying its predictions:
![Simple Logistic Regression](./visuals/simple_logreg_cm.png)
You can see that it predicted quite well for classes 0 and 2, but struggled predicting accurately for class 1 (functional wells that need repair). This was in part due to the very small sample size of class 1 (7% of our dataset).
3. To address the imbalanced class sizes, I incorporated SMOTE (synthetic minority oversampling). Same as the above model, I used a default logistic regression but this time with SMOTE:
   - SMOTE oversamples all minority classes, and uses k-nearest neighbors to produce synthetic data.
   - In the basic SMOTE model, the predictions for class 1 were indeed more accurate. However, the accuracies of both class 0 and class 2 declined. The overall prediction accuracy decreased to 64%.
  
## More advanced models



# Repo ToC:
<br>
  - [Data cleaning notebook](./cleaning.ipynb)
  - [Data analysis notebook](./analysis.ipynb)
  - [Final model notebook](./model_pipeline.ipynb)
  - [Description of features](./features.txt)
