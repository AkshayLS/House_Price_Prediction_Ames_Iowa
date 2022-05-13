import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import shap
st.set_option('deprecation.showPyplotGlobalUse', False)


st.write("""
# Ames House Price Prediction App
We are using the Ames dataset for prediction. This app predicts the sale price of residential homes in Ames, Iowa.
""")


st.write('---')
st.write('### Description of some Features in the Dataset')
st.write('This dataset actually contains 79 explanatory variables but we will be using only the top 10 features which'
         ' affect the target variable(SalePrice). You can get more information about the top 10 features by checking out my'
         ' github repository([House_Price_Prediction_Ames_Iowa](https://github.com/AkshayLS/House_Price_Prediction_Ames_Iowa)).'
         ' The features we will be using are as follows: ')
st.write('**GrLivArea** - Above grade (ground) living area square feet')
st.write('**OverallQual** - Rates the overall material and finish of the house')
st.write('**TotalBsmtSF** - Total square feet of basement area')
st.write('**LotArea** - Lot size in square feet')
st.write('**1stFlrSF** - First Floor square feet')
st.write('**YearBuilt** -  Original construction date')
st.write('**BsmtFinSF1** - Type 1 finished square feet')
st.write('**YearRemodAdd** - Remodel date (same as construction date if no remodeling or additions)')
st.write('**GarageArea** - Size of garage in square feet')
st.write('**Fireplaces** - Number of fireplaces')
st.write('**SalePrice** - Sale Price of the House')
st.write('---')


data = pd.read_csv('data.csv',usecols=['GrLivArea','OverallQual','TotalBsmtSF','LotArea','1stFlrSF','YearBuilt','BsmtFinSF1','YearRemodAdd','GarageArea','Fireplaces','SalePrice'])
st.write("""
### Preview of the dataset with only the top 10 features
""")
st.write(data)


#Visualisation
st.write('### Explore Data')
chart_select = st.sidebar.selectbox(
    label ="Type of chart",
    options=['Scatterplot','Barplot','Lineplot','Histogram','Boxplot']
)

numeric_columns = list(data.select_dtypes(['float','int']).columns)

if chart_select == 'Scatterplot':
    st.sidebar.subheader('Scatterplot Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
        st.write('#### Scatter Plot')
        plot = px.scatter(data_frame=data,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Barplot':
    st.sidebar.subheader('Barplot Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
        st.write('#### Bar Plot')
        plot = px.bar(data_frame=data,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Histogram':
    st.sidebar.subheader('Histogram Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        st.write('#### Histogram')
        plot = px.histogram(data_frame=data,x=x_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Lineplot':
    st.sidebar.subheader('Lineplot Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
        st.write('#### Line Plot')
        plot = px.line(data,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Boxplot':
    st.sidebar.subheader('Boxplot Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        st.write('#### Box Plot')
        plot = px.box(data,x=x_values)
        st.write(plot)
    except Exception as e:
        print(e)
st.write('---')


# Feature Engineering
data1=data.copy()
X=data1.drop(columns='SalePrice', axis=1)
Y=data1.SalePrice
feature_list_X=[feature for feature in X.columns]
for feature in feature_list_X:
    if feature == "GrLivArea" :
            X.loc[(X.GrLivArea > 4000), 'GrLivArea'] = np.median(X.GrLivArea)
            X[feature]=np.log1p(X[feature])
    elif feature == 'OverallQual' :
            X[feature]=X[feature].replace({1 : 1, 2 : 1, 3 : 1,
                                           4 : 2, 5 : 2, 6 : 2,
                                           7 : 3, 8 : 3, 9 : 3, 10 : 3})
    elif feature == 'TotalBsmtSF' :
            X.loc[(X.TotalBsmtSF > 6000), 'TotalBsmtSF'] = np.median(X.TotalBsmtSF)
            X[feature]=np.log1p(X[feature])
    elif feature == 'LotArea' :
            X[feature]=np.log1p(X[feature])
    elif feature == '1stFlrSF' :
            X[feature]=np.log1p(X[feature])
    elif feature == 'YearBuilt' :
            X[feature]=np.log1p(X[feature])
    elif feature == 'BsmtFinSF1' :
            X[feature]=np.log1p(X[feature])
    elif feature == 'YearRemodAdd' :
            X[feature]=np.log1p(X[feature])
    elif feature == 'GarageArea' :
            X.loc[(X.GarageArea > 1200), 'GarageArea'] = np.median(X.GarageArea)
    elif feature == 'Fireplaces' :
            X[feature]=np.log1p(X[feature])

sc=StandardScaler()
feature_cat_X=['OverallQual']
feature_num_X=[feature for feature in feature_list_X if feature not in feature_cat_X]
X.loc[:,feature_num_X] = sc.fit_transform(X.loc[:,feature_num_X])
Y=np.log1p(Y)


# Creating input sliders for feature Variables
st.sidebar.header('Specify Input (Feature Variables): ')
def user_input_features():
    GrLivArea = st.sidebar.slider('Ground Living Area', int(data['GrLivArea'].min()), int(data['GrLivArea'].max()), int(data['GrLivArea'].mean()))
    OverallQual = st.sidebar.slider('Overall House Quality', int(data['OverallQual'].min()), int(data['OverallQual'].max()),int(data['OverallQual'].mean()))
    TotalBsmtSF = st.sidebar.slider('Total Basement sqft', int(data['TotalBsmtSF'].min()),
                                    int(data['TotalBsmtSF'].max()),int(data['TotalBsmtSF'].mean()))
    LotArea  = st.sidebar.slider('Lot Area', int(data['LotArea'].min()), int(data['LotArea'].max()),int(data['LotArea'].mean()))
    FirstFlrSF    = st.sidebar.slider('1st Floor sqft', int(data['1stFlrSF'].min()), int(data['1stFlrSF'].max()),int(data['1stFlrSF'].mean()))
    YearBuilt = st.sidebar.slider('Year Built', int(data['YearBuilt'].min()),
                                    int(data['YearBuilt'].max()),int(data['YearBuilt'].mean()))
    BsmtFinSF1 = st.sidebar.slider('Type 1 finished square feet', int(data['BsmtFinSF1'].min()),
                                    int(data['BsmtFinSF1'].max()),int(data['BsmtFinSF1'].mean()))
    YearRemodAdd = st.sidebar.slider('Remodelled Year', int(data['YearRemodAdd'].min()),
                                    int(data['YearRemodAdd'].max()),int(data['YearRemodAdd'].mean()))
    GarageArea = st.sidebar.slider('Garage Area', int(data['GarageArea'].min()), int(data['GarageArea'].max()))
    Fireplaces = st.sidebar.slider('Number of fireplaces', int(data['Fireplaces'].min()),
                                    int(data['Fireplaces'].max()),int(data['Fireplaces'].mean()))
    datax = {'GrLivArea': GrLivArea,'OverallQual': OverallQual,'TotalBsmtSF': TotalBsmtSF,'LotArea': LotArea,'1stFlrSF':FirstFlrSF,
             'YearBuilt':YearBuilt,'BsmtFinSF1':BsmtFinSF1,'YearRemodAdd':YearRemodAdd,'GarageArea': GarageArea,'Fireplaces':Fireplaces}
    final_df = pd.DataFrame(datax, index=[0])
    st.write('### Specified Input Parameters')
    st.write(final_df)
    feature_list=[feature for feature in final_df.columns]
    for feature in feature_list:
        if feature == "GrLivArea" :
            final_df.loc[(final_df.GrLivArea > 4000), 'GrLivArea'] = np.median(data.GrLivArea)
            final_df[feature]=np.log1p(final_df[feature])
        elif feature == 'OverallQual' :
            final_df[feature]=final_df[feature].replace({1 : 1, 2 : 1, 3 : 1,
                                                       4 : 2, 5 : 2, 6 : 2,
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3})
        elif feature == 'TotalBsmtSF' :
            final_df.loc[(final_df.TotalBsmtSF>6000),'TotalBsmtSF'] = np.median(data.TotalBsmtSF)
            final_df[feature]=np.log1p(final_df[feature])
        elif feature == 'LotArea' :
            final_df[feature]=np.log1p(final_df[feature])
        elif feature == '1stFlrSF' :
            final_df[feature]=np.log1p(final_df[feature])
        elif feature == 'YearBuilt' :
            final_df[feature]=np.log1p(final_df[feature])
        elif feature == 'BsmtFinSF1' :
            final_df[feature]=np.log1p(final_df[feature])
        elif feature == 'YearRemodAdd' :
            final_df[feature]=np.log1p(final_df[feature])
        elif feature == 'GarageArea' :
            final_df.loc[(final_df.GarageArea > 1200), 'GarageArea'] = np.median(data.GarageArea)
        elif feature == 'Fireplaces' :
            final_df[feature]=np.log1p(final_df[feature])

    feature_cat=['OverallQual']
    feature_num=[feature for feature in feature_list if feature not in feature_cat]
    # Scale
    final_df.loc[:,feature_num] = sc.transform(final_df.loc[:,feature_num])
    return final_df



dy = user_input_features()


# Print Specified Input Features
#st.header('Specified Input Parameters After Transformation')
#st.write(dy)
#st.write('---')


# Train model
model = CatBoostRegressor()
model.fit(X, Y)


# Predict Sale Price
y_predict = model.predict(dy)
original_predict_value=np.floor(np.exp(y_predict)-1)
st.write('### Prediction of House Price')
st.write(
        f"""
        ##### Predicted house price with given input parameters in USD is : {original_predict_value[0]} ! 
        """)


# Explaining the model's predictions using SHAP values
st.write('### Feature Importance')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
if st.button('Show SHAP Graphs'):
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X)
    plt.gcf().axes[-1].set_aspect('auto')
    plt.tight_layout()
    # Smaller "box_aspect" value to make colorbar thicker
    plt.gcf().axes[-1].set_box_aspect(50)
    st.pyplot(bbox_inches='tight')
    st.write('---')
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(bbox_inches='tight')
