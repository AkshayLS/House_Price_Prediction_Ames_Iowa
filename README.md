
# House Price Prediction - Ames,Iowa
It is a Advanced Problem of Regression which requires advanced techniques of feature engineering, feature selection, modelling building and model evaluation.
We will be using the [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) taken from a kaggle competition.
We will only be using the train data from that competition.

# Description
Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset. 
For more details about the features please read the data description file.

# Goal
We do not have accurate estimation of house prices even though there is a lot of data present around us and also available to us. Proper and justified prices of properties can bring in a lot of transparency in the real estate industry which is important for consumers.
The goal of this project is to predict the final price of each home based on several variables describing almost every aspect of the house and its surroundings.

# Code 
Please refer the ipynb notebooks for a detailed walkthrough of the entire project.

# Workflow
1.  Exploratory Data Analysis   
    - Basic Checks
    - Missing Values
    - Year Feature
    - Numerical Features
    - Categorical features
    - Target Variable
    - Outliers
    - Correlation
    - Multicollinearity
2.  Pre-processing
    - Treating Outliers
    - Transform Target Variable
    - Missing Value Imputation
    - Correcting Skewness
    - Encoding
    - Train Test Split
    - Scaling
    - Feature Selection
3.  Model Building and Evalation  
    - Models
    - Performance Table
    - Hyper tune few Models
    - Hyper tuned model Performance table
    - Feature Importance
    - Reversing back Target Variable to Original Value

# Evalation Metric
Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted price value and the logarithm of the observed sales price.
Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.

# Web App
I have built a streamlit web app to predict the sale price using only the top 10 features from our final model.
It can be found here https://share.streamlit.io/akshayls/house_price_prediction_ames_iowa/main/HPwebapp.py
Steps to run this app in local machine :

```
pip install -r requirements.txt 
```
then
```
streamlit run streamlit_app.py
```

# Contributers
Akshay Shettigar

# Acknowledgments
1.  Inspirations are drawn from various Kaggle notebooks but majorly motivation is from the following :
    - [A study on Regression applied to the Ames dataset](https://www.kaggle.com/code/juliencs/a-study-on-regression-applied-to-the-ames-dataset)
    - [Beginners_Prediction_Top3%](https://www.kaggle.com/code/marto24/beginners-prediction-top3)
    - [A Detailed Regression Guide with House-pricing](https://www.kaggle.com/code/masumrumi/a-detailed-regression-guide-with-house-pricing)
2.  For Deployment :
    - [Data Professor Boston House Price Prediction Web App](https://www.youtube.com/watch?v=z5HfbXORZsg)
    - [Machine Learning Model Deployment Using Streamlit](https://www.youtube.com/watch?v=jL2ZRkSopBg)
