# Project Description
Kaggle Competition - https://www.kaggle.com/c/house-prices-advanced-regression-techniques

Using the training dataset consisting of 79 columns and sale prices in order to predict the sale prices on the test dataset.

Scoring metric was RMSLE (Root Mean Squared Log Error).

# Feature Clean Up/Engineering
Replaced Missing Values with 0, 'NONE', Median, Mode

Removed Outliers

Created ratio values (Living Space/Lot Area)

Changed years to years since (2014 becomes 5)

Logged and Squared Columns

# Models
## Random Forest
Produced a feature importance list that was used in the other models. In the other models, the number of columns to drop (dropcolumns) was treated as a hyperparameter. For example, dropcolumns = 50, would mean that the 50 least important columns were excluded from the model. This lead to significant increases in model accuracies.

RMSLE = 0.146

## XGBoost
RMSLE = 0.130

## Linear 
Target Variable (Sale Price) was logged.

dropcolumns = 217

RMSLE = 0.11748

## Lasso
Target Variable was logged.

dropcolumns = 204

RMSLE = 0.11718

## Support Vector (Polynomial)
C = 200, Degree = 6, Epsilon = 100, Coef0 = 2

RMSLE = 0.12285

## Neural Network
Standard scaling for input variables.
Target variable was logged.

Hidden Layer (20 units, Linear)

Dropout Layer (0.5)

Hidden Layer (20 units, Linear)

Dropout Layer (0.5)

220 Epochs

RMSLE = 0.12770

## Stacked Model
Weighted average of all the models according to each model's RMSLE

Weight = 1/(RMSLE^4)

RMSLE = 0.11512

## Results
92nd Percentile

