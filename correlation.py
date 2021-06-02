import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz
from IPython.display import Image  
from sklearn import tree
##import graphviz
from matplotlib import pyplot as plt
import seaborn as sns
##from statsmodels.graphics.mosaicplot import mosaic
import scipy
import missingno as msno
import awesome_streamlit as ast

df = pd.read_csv('Bank_CS.csv')
df.drop(columns=['Unnamed: 0','Unnamed: 0.1'], inplace=True)

numeric_features = df.select_dtypes(include=[np.number])
data = numeric_features.copy()
_ = df['Decision'].map(lambda x: 1 if x == 'Reject' else 0)
data['Decision'] = _
correlation_num = data.corr()

categorical_features = df.select_dtypes(include=[np.object])
data = categorical_features.copy()
data = data.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
correlation_cat = data.corr()

def write():
    with st.spinner("Loading EDA ..."):
        ast.shared.components.title_awesome("Exploratory Data Analysis")
    
    st.write(
        """
    # Correlation Analysis

    """
    )
    st.subheader("Skewness and Kurtosis")
    skew = df.skew()
    kurt = df.kurt()
    st.markdown("Skew")
    st.write(skew)
    st.markdown("Kurt")
    st.write(kurt)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader("Skewness of the Dataset")
    sns.distplot(df.skew(),color='blue',axlabel ='Skewness')
    st.pyplot()
    st.markdown("The dataset is not normally distributed.")

    st.subheader("Correlation of Numeric Features")
    f , ax = plt.subplots(figsize = (14,12))
    plt.title('Correlation of Numeric Features with Decision',y=1,size=16)
    sns.heatmap(correlation_num,square = True,  vmax=0.8)
    st.pyplot()
    st.markdown("The **_number of properties_** has a slight positive correlation with **_number of dependents_**, **_years to financial freedom_** and **_number of credit card facilities_**.")
    st.markdown("**_Number of bank products_** also has a slight positive correlation with **_years to financial freedom_**.")

    st.subheader("Correlation of Categorical Features")
    f , ax = plt.subplots(figsize = (14,12))
    plt.title('Correlation of Categorical Features with Decision',y=1,size=16)
    sns.heatmap(correlation_cat,square = True,  vmax=0.8)
    st.pyplot()
    st.markdown("The **_employment type_** is positively correlated with the **_state_** and **_property type_**.")
    st.markdown("There is no obvious correlation between the features and the dependent variable.")

if __name__ == "__main__":
    write()