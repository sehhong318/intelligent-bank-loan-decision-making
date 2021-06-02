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
import awesome_streamlit as ast

df = pd.read_csv('Bank_CS.csv')
df.drop(columns=['Unnamed: 0','Unnamed: 0.1'], inplace=True)
numeric_features = df.select_dtypes(include=[np.number])
year_feature = [feature for feature in numeric_features if 'Month' in feature or 'Year' in feature]
year_feature.remove('Monthly_Salary')
## get discrete features
discrete_feature=[feature for feature in numeric_features if len(df[feature].unique())<25]
discrete_feature = list(set(discrete_feature) - set(year_feature))
continuous_feature=[feature for feature in numeric_features if feature not in discrete_feature+year_feature]

def write():
    with st.spinner("Loading EDA ..."):
        ast.shared.components.title_awesome("Exploratory Data Analysis")
    
    st.write(
        """
    # Continuous Features

    """
    )
    st.markdown("The histogram plots allow us to observe the distribution of the continuous variables.")
    for feature in continuous_feature:
        fig = plt.figure(figsize = (10,7))
        ax = fig.gca()
        data=df.copy()
        data[feature].hist(bins=50, ax=ax)
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.title(feature)
        #plt.show()
        st.pyplot(plt)

    st.subheader("Observations: ")
    st.markdown('---')
    st.markdown('**How do the continuous variables distributed across the raw data?**')
    st.markdown("All the continuous features of this dataset are not normally distributed.")

if __name__ == "__main__":
    write()