# Machine Learning for Life Sciences Project 1 - Hourly Bicyclist Prediction on Coupure Links, Ghent
This project focuses on predicting hourly bicyclist counts on Coupure Links in Ghent, employing a Histogram Gradient Boosting regressor to forecast July values based on data from January to June, while also serving as the initial foray into essential machine learning practices within the Machine Learning for Life Sciences course at Ghent University. The project was performed in Google Colab notebook in Python running on CPU.

## Introduction

In this project, our goal is to predict the hourly bicyclists passing on Coupure Links in Ghent. First a thorough exploration of the dataset, encompassing descriptive statistics and visualizations to discern patterns, trends, and potential outliers, is performed. Feature selection and pre-processing follow, including handling missing values, encoding categorical variables, and considering temporal aspects. After pre-processing the data, a Histogram Gradient Boosting regressor model is used to predict the values of bicyclist passing by in July based on data of january-june.

On top of the primary forecasting goal, this project also acts as a first start to good machine learning practice including data exploration, validation of techniques used and an overall understanding of how to run a machine learning project from start to end.

## Data Exploration

### First Dataset

The project begins with the exploration of hourly bike counts from January 1st to June 30th. Anomalies, particularly a data gap indicating zero values, are identified and addressed by removing corresponding entries. Despite the removal, challenges arise due to missing data during the Easter break, a period crucial for July predictions. The dataset's right-skewed distribution, resembling a Poisson distribution, poses considerations for model training.

### Second Dataset

The second dataset, encompassing weather-related variables, undergoes thorough exploration. Correlations prompt the removal of redundant variables, and categorical features are extracted from the 'time' column to enhance the dataset. Notably, the dataset's correlation structure highlights relationships critical for accurate predictions.

## Data Preprocessing

Data points with no bike counts and redundant variables are removed, ensuring consistency between datasets. An omitted data point during the transition from winter to summer time is addressed. StandardScaler is applied to non-categorical variables for model robustness, preparing the datasets for model training.

## Model Training and Validation

HistGradientBoostingRegressor: Selected for efficiency and resistance to outliers, this model undergoes fine-tuning through GridSearchCV. Optimized hyperparameters, including loss function, maximum iterations, tree depth, learning rate, and L2 regularization, contribute to the model's predictive capabilities.

## Conclusion

This project provides valuable insights into machine learning workflows, emphasizing the significance of data exploration, preprocessing, and model selection. Challenges, including anomalies and predicting an abnormal July, underscore the importance of careful data curation. Despite a trial-and-error approach, achieving a satisfactory prediction score on Kaggle validates the project's success, contributing to an enhanced understanding of machine learning techniques and best practices in data science.
