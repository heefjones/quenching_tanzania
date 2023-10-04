# Overview:

Given various features of water wells in Tanzania, predict each well's condition/functionality. The possible values for well functionality are as follows:

  - Functional (class 0)
  - Functional, Needs repairs (class 1)
  - Non-functional (class 2)

The end goal is to build a predictive model that properly identifies each of these 3 classes with maximum accuracy.

### Goals:

- Identify the features that correlate most with water well functionality.
- Create a Logistic Regression model that predicts a water well's condition with maximum accuracy.



# Data

The [data](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/) consisted of close to 60,000 water wells in Tanzania, recorded by GeoData Consultants Ltd. The data contained the well's type, water source, geographical info, as well as payment information and the organizations or governmental bodies involved.



# Data Cleaning
To begin, each well had 39 corresponding columns that described unique information for each well. I scoured through this data and removed columns that held redundant information. I also condensed columns with many unique values. For example, there were almost 2000 unique organizations that funded the building of the well. I kept 91 funders that funded 100+ wells in the data, as they made up 73% of the dataset. The remaining 27% were categorized as "other". I preformed a similar process for the "installer" column, taking only the installers that had installed 100+ wells. This was to minimize the amount of encoded columns that my models had to parse through. Overall, this made the modeling process smoother and led to faster convergence.

Talk about cleaining columns where 0 was a placeholder for missing values

Only 7 of the columns contained null values



# Modeling
I began the modeling process by using the 21-feature data. After plateauing with a maximum accuracy of 74.8%, I ran the 23-feature through a default Logistic Regression and recorded a new-best accuracy of 77.2%. For the rest of the modeling process, I used the 23-feature dataset. I realize that removing those 2 extra features cost my model several accuracy points. In hindsight, it seems foolish to experiment with less features, rather than starting with more and removing features throughout the process. For future projects, I plan on only removing features after thoroughly testing the larger feature set.

Here are the accuracy results from the first 3 [simple models](./simple_models.ipynb):
1. The baseline model (sklearn.DummyClassifier):
   - The baseline model simply predicted the most common class (functional), and predicted with 54.2% accuracy.
2. A simple logistic regression (sklearn.LogisticRegression):
   - This model used default parameters and predicted with 77.2% accuracy. Here is a confusion matrix displaying its predictions:
![Simple Logistic Regression](./visuals/simple_logreg_cm.png)
You can see that it predicted quite well for classes 0 and 2, but struggled predicting accurately for class 1 (functional wells that need repair). This was in part due to the very small sample size of class 1 (7% of our dataset).
3. To address the imbalanced class sizes, I incorporated SMOTE (synthetic minority oversampling). Same as the above model, I used a default logistic regression but this time with SMOTE:
  - SMOTE oversamples all minority classes, and uses k-nearest neighbors to produce synthetic data.
  - In the basic SMOTE model, the predictions for class 1 were indeed more accurate. However, the accuracies of both class 0 and class 2 declined. The overall prediction accuracy decreased to 64%.
  
## More advanced models



# Repo ToC:

- [Data cleaning notebook](./cleaning.ipynb)
- [Data analysis notebook](./analysis.ipynb)
- [Final model notebook](./model_pipeline.ipynb)
- [Description of features](./features.txt)
