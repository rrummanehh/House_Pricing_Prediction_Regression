House Price Prediction — Ames Housing Dataset

Overview

This project predicts residential house sale prices using the Ames Housing dataset. It goes through the full machine learning pipeline — cleaning the data, engineering new features, encoding categorical variables, training multiple regression models, tuning them with GridSearchCV, and comparing their performance.

This was my first regression task, completed during my second year of university. It was also my second machine learning project overall, and having done a classification project before it helped me come into this one with a better understanding of the workflow.
---------------------------------------

Dataset

The Ames Housing dataset contains detailed information about residential properties in Ames, Iowa. It includes structural features, quality ratings, neighborhood data, garage info, basement details, and sale prices.

To run the notebook locally, keep AmesHousing.csv inside the folder named data/, just like you downloaded it from the repository. If you are using Google Colab, place the file in the root directory and uncomment the Colab path at the top of the notebook.
---------------------------------------

What I Did

1. Data Cleaning

. Dropped columns with too many missing values or no predictive value (Pool QC, Alley, Fence, Order, PID, etc.)
. Filled missing numerical values with the median
. Filled missing categorical values with the mode
. Basement columns with no basement were filled with 'none' instead of leaving them null
. Fixed negative house age values caused by data inconsistencies

2. Feature Engineering

New features were created to reduce the number of raw columns and capture more meaningful patterns:

. total SF — total living area across all floors and basement
. total bathrooms — weighted count of all bathrooms
. house age — how old the house was when it was sold
. remodeled — 1 if the house was remodeled after it was built, 0 otherwise
. average SF per room — average space per room
. Quality Index — overall quality score multiplied by overall condition
. Lot Utilization — ratio of living space to total lot size

3. Encoding

. Ordinal Encoding for features with a meaningful order, such as quality ratings (Po, Fa, TA, Gd, Ex)
. One-Hot Encoding for nominal features like neighborhood, house style, and sale condition

4. Scaling

StandardScaler was applied to the training set and used to transform the test set. Scaling was only applied for KNN, Linear Regression, Ridge, and Lasso. Decision Tree and Random Forest do not need it.

5. Model Training and Tuning

All models were tuned using GridSearchCV with scoring='neg_root_mean_squared_error'. The best parameters from each search are saved as comments in the notebook for reference, and the final models use those parameters directly.
---------------------------------------

A Note on Data Leakage

During feature importance analysis, the Decision Tree was only using a single feature and had an R² of 0.9685, which was suspiciously high. After investigating, it turned out that a feature called Price Per SF that i created by dividing SalePrice by total SF. Since SalePrice is the target variable, this feature was essentially leaking the answer into the input. After removing it and retraining all models, the scores dropped but became honest and trustworthy.
---------------------------------------

Results:

 Model	                Test RMSE       Train RMSE      R²
. Random Forest 	27,485	        20,052	        0.8964
. Lasso	                28,664	        22,583	        0.8873
. Linear Regression	28,943	        22,476	        0.8851
. Ridge	                30,355	        24,890	        0.8736
. Decision Tree	        33,013   	27,963	        0.8505
. KNN	                37,184	        29,031 	        0.8103

Random Forest had the lowest test RMSE and the highest R². The linear models performed surprisingly close to it. KNN was the weakest model across the board.
---------------------------------------

Visualizations

Three sets of plots are generated at the end of the notebook:

. RMSE and R² Comparison — side-by-side bar charts for all six models.
. Actual vs Predicted — scatter plots showing how close each model's predictions are to the real values, with a perfect fit reference line.
. Feature Importance — top 10 features per model. Tree models use built-in feature importance, linear models use absolute coefficient values, and KNN uses correlation with the target variable since it has no native importance measure.

---------------------------------------