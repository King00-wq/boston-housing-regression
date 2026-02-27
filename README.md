# boston-housing-regression
Machine learning regression project predicting house prices using the Boston Housing dataset. Includes EDA, preprocessing, model training, and performance evaluation. Demonstrates practical understanding of regression concepts and model comparison techniques.

Boston Housing Regression:

Machine learning regression project for predicting house prices using the Boston Housing dataset. This project includes exploratory data analysis, preprocessing, model training, and performance evaluation. The aim is to understand regression concepts practically and compare different models.

Project Overview:

This project focuses on building regression models to predict the median value of houses based on different features like crime rate, number of rooms, tax rate, etc.

The dataset used is the Boston Housing dataset, introduced in the research paper:

Harrison, D., & Rubinfeld, D. L. (1978). Hedonic housing prices and the demand for clean air. Journal of Environmental Economics and Management, 5(1), 81–102.

The dataset contains 506 rows and 13 input features. The target variable is MEDV (Median value of owner-occupied homes in $1000s).

Dataset Features:

CRIM – Per capita crime rate by town
ZN – Proportion of residential land zoned for large lots
INDUS – Proportion of non-retail business acres
CHAS – Charles River dummy variable
NOX – Nitric oxide concentration
RM – Average number of rooms per dwelling
AGE – Proportion of units built before 1940
DIS – Distance to employment centers
RAD – Accessibility to highways
TAX – Property tax rate
PTRATIO – Pupil-teacher ratio
B – Racial proportion related variable
LSTAT – Percentage of lower status population

Ethical Note:

The dataset has been removed from some libraries like scikit-learn because one of the variables (B) is related to race. It is still useful for learning regression concepts, but in real-world projects, ethical data handling is important.

Objectives:

The main objectives of this project were:

• Perform exploratory data analysis
• Understand relationships between features and target
• Preprocess the dataset
• Train multiple regression models
• Compare model performance
• Interpret results

Exploratory Data Analysis

During EDA, the following steps were performed:

• Checked summary statistics
• Observed distributions of features
• Plotted correlation matrix
• Identified important features

From correlation analysis, RM (number of rooms) had a strong positive correlation with price, while LSTAT had a strong negative correlation.

Data Preprocessing:

The following preprocessing steps were applied:

• Train-test split
• Feature scaling using StandardScaler
• Checking for multicollinearity

Scaling was important because some models perform better when features are on the same scale.

Models Used

The following regression models were implemented:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Each model was evaluated using:

• Mean Absolute Error (MAE)
• Mean Squared Error (MSE)
• Root Mean Squared Error (RMSE)
• R² Score

Results

Linear Regression worked as a good baseline model.
Decision Tree captured non-linear patterns but showed signs of overfitting.
Random Forest gave better overall performance compared to individual models.

Technologies Used

Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn

Project Structure:

boston-housing-regression/
│
├── data/
│   └── boston.csv
│
├── notebooks/
│   └── boston_analysis.ipynb
│
├── app/
│   └── app.py
│
├── models/
│   └── model.pkl
│
├── requirements.txt
│
├── README.md
├── LICENSE
└── .gitignore

Conclusion:

This project helped in understanding the complete regression workflow from data analysis to model evaluation. It also helped in learning how different regression models behave on the same dataset and how to compare them properly.

Future Improvements

• Hyperparameter tuning
• Cross-validation
• Regularization techniques like Ridge and Lasso
• Deploying the model using Flask or FastAPI

This project represents my practical implementation of regression concepts as a second-year B.Tech CSE student.
