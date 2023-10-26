PfizerAssignment
==============================
# Please check code in either main.py or main.ipynb

### The following study is divided into 5 Major steps
#### Step-1) Data Import and preprocessing
#### Step-2) Exploratory Data Analysis, Data Visualization and Feature Engineering
#### Step-3) Model Selection and Implementation 
#### Step-4) Model Evaluation and Performance Measurement
#### Step-5) Conclusions

#### Step-1: Data Import and Preprocessing
In the first step all the required packages for the model development are imported and data is downloaded from excel file into a dataframe. The data dictionary is also downloaded for reference

Preliminary data investigation activities such as data quality check, identification of null or missing data, inaccurate data check and renaming of features are done in this step

In the sample dataset, no missing values were identified.

There are five instances where the major vessels colored by fluoroscopy (ca) have a value of 4, whereas the acceptable range is 0-3.

Additionally, two instances have a Thalassemia value of zero, although the valid range is 1-3.

Moreover, the "chest pain type" variable contains unique values ranging from 0 to 3, but the data dictionary lacks descriptions for these numerical codes.
The dataset comprises four binary input variables, one of which is the target value.

#### Step-2: Exploratory Data Analysis, Data Visualization and Feature Engineering
This step involves examining and understanding the characteristics and patterns within a dataset before applying any formal modeling techniques. Generating descriptive statistics and understanding the central tendency and dispersion of the data. 

Visualizations like histograms, scatter plots, and heatmaps are used to visualize the distribution, relationships, and patterns in the data. This aids in identifying outliers and understanding the structure of the data. EDA explores the relationships between different variables. Correlation analysis helps identify which variables are strongly or weakly correlated with each other.

1) The dataset is imbalanced with respect to gender; 207 observations for male and only 96 observations for female

2) Cholesterol levels shows almost normalized across the population mean. There are one outliers where cholesterol level > 500 with confirmed heart disease

3) One person with age 67 is found be with less than < 80 heart rate

4) ST Depression Induced by Exercise is severly right skewed with 118 values being 0.0

6) Chest_pain and Max_Heart_Rate are highly positively correlated to Heart Disease whereas ST Depression Induced by Exercise and 
   Exercise induced agina are highly negatively correlated (Threshold = 0.3)
   
### Step-2B: Feature Engineering
In this step, feature engineering is performed to identify and select a subset of the most relevant features (variables) from the original set of feature. This is done to improve the performance of the classification model, reduce computational complexity, and avoid overfitting

Calculate the correlation between each feature and the target variable and then use the correlation coefficients to find the highly impacted features. But in this case there are two methods that are used to select the features

1) Recursive Feature Elimination (RFE): Train the classification model on the entire set of features and rank the features based on their importance scores. Remove the least important features and repeat the process until the desired number of features is reached.

2) L1 Regularization (Lasso): Apply L1 regularization to linear models (e.g., Logistic Regression).This method penalizes the absolute magnitude of coefficients, encouraging some of them to be exactly zero. The non-zero coefficients correspond to selected features.

Create Input Dataset by Applying Standard Scaler

The purpose of the Standard Scaler in machine learning is to standardize or normalize the features in a dataset. 
Standardization is an essential preprocessing step in many machine learning algorithms that are sensitive to the scale of the features. 

The dataset contains Resting_BP, Cholesterol, Age, Max_Heart_Rate values that are far higher than values of other variables. So Standardization helps in achieving better model performance and can make optimization algorithms converge faster.

Since there are a very few outliers Standard Scaler can be applied to this dataset

### Recursive Feature Elimination
In the process of Recursive Feature Elimination with cross-validation, four different estimators (Logistic Regression, Random Forest, Decision Tree, and Gradient Boosting) were employed. The resulting ranks, accuracy levels, and selected features were recorded in a dataframe.

Interestingly, both the Logistic Regression model with a Decision Tree classifier and the Decision Tree model with the same classifier achieved a remarkable accuracy of 100% using 12 features. This indicates a tie between the two approaches, suggesting that Logistic Regression or Decision Tree with a Decision Tree classifier could be considered the preferred method.

#### Step-3: Model Selection and Implementation
##### 1) Decision Tree Classifier
We'll begin by assessing the model using a decision tree classifier. We'll perform hyperparameter tuning for the max_depth to find the model that achieves the highest area under the curve. Once the optimal max_depth is determined, it will be employed for training and testing the model. After the Decision Tree Classifier, Logistic Regression Classifier will be trained

Drop 'age', 'resting_bp','cholesterol', 'fast_blood_sugar', 'rest_ecg' features from input
##### Perform Max Depth Determination

##### Hyperparameter max_depth determination

Training and testing the decision tree classifier will be done using max_depth parameter values ranging from 1 to 10. For this process, 80% of the input data will be utilized for training, while the remaining 20% will be reserved for testing. To ensure reproducibility, random state is set to 42

##### Max-Depth = 4 has the highest accuracy
Max_depth value 4 is found to have highest accuracy based on the hyperparameter tuning. Run model for the max_depth value of 4 to find the final accuracy of Decision Tree Classifier

##### 2) Logistic Regression Classifier

Perform training on Logistic Regression Classifier. Drop 'age', 'resting_bp','cholesterol', 'fast_blood_sugar', 'rest_ecg' columns before running the model
##### 3)  Logistic Regression Classifier with Lasso Regularization

Lasso regularization, also known as L1 regularization, is a technique used in machine learning and statistics to add a penalty term based on the absolute value of the coefficients in a linear model. This encourages the model to select a subset of the most important features, effectively performing feature selection and reducing overfitting.

##### Conclusions
1) Decision Tree Classifier
2) Logistic Regression Classifier:
3) Logistic Regression with Lasso Regularization:


a) The Decision Tree Classifier, Logistic Regression Classifier, and Logistic Regression with Lasso Regularization all exhibit a False Positive Rate (FPR) of 13.7% and a True Negative Rate (TNR) of 12.5%.

Given that all three models share the same FPR/TNR rates, the model with the highest Accuracy is considered the optimal choice. Therefore, any of the Logistic Regression Models can be selected as the final model (with accuracy of 87%). Moreover, they boast the highest Recall Rate (88%), a critical factor in medical disease recognition.

b) All These are Contributing Factors based on the Coefficients determined by logistic regression

    Chest Pain
    Sex
    Vessels_Colored
    ST Depression induced by Exercise
    Thalassemia
    Exercise Induced Angina
    Slope of Peak Exercise ST Segment
    Max Heart Rate


==============================

This Project is developed based on the requirements of Pfizer take home assignment

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
