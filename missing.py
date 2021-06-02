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
from PIL import Image
import awesome_streamlit as ast

df = pd.read_csv('Bank_CS.csv')
df.drop(columns=['Unnamed: 0','Unnamed: 0.1'], inplace=True)

def write():
    with st.spinner("Loading EDA ..."):
        ast.shared.components.title_awesome("Exploratory Data Analysis")
    
    st.write(
        """
    # Missing Values Analysis

    """
    )

    st.subheader("Matrix of Missing Values in Raw Dataset")
    #g = msno.matrix(df.sample(500))
    #st.pyplot(msno.matrix(df.sample(500)))
    #msno.matrix(df.sample(500))
    img = Image.open('img/Bank_EDA_cell_23_output_1.png')
    st.image(img, use_column_width=True)

    st.subheader("Barplot of Missing Values in Raw Dataset")
    # g = msno.bar(df.sample(1000))
    # st.pyplot(g)
    img = Image.open('img/Bank_EDA_cell_24_output_1.png')
    st.image(img, use_column_width=True)

    st.markdown("By plotting a matrix and a barplot on missing values, the portion of missing values can be visualized.")

    st.subheader("Correlation Heatmap of Missing Values")
    st.markdown("To observe whether there are any correlations between the missing features, a heatmap can be plotted.")
    # g = msno.heatmap(df)
    # st.pyplot(g)
    img = Image.open('img/Bank_EDA_cell_26_output_1.png')
    st.image(img, use_column_width=True)
    st.markdown("There is no significant correlation between the presence or absence of the variables.")

    st.subheader("Dendogram of Missing Values")
    st.markdown("To confirm this hypothesis, a dendrogram is plotted.")
    # g=msno.dendrogram(df)
    # st.pyplot(g)
    img = Image.open('img/Bank_EDA_cell_28_output_1.png')
    st.image(img, use_column_width=True)
    st.markdown("There are many fields which need imputation, but however, the existing variables do not predict each other well.")

    st.subheader("Missing Categorical Features")
    categorical_features = df.select_dtypes(include=[np.object])
    total = categorical_features.isnull().sum().sort_values(ascending=False)
    percent = ((categorical_features.isnull().sum()/categorical_features.isnull().count())*100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1,join='outer', keys=['Total Missing Count', ' % of Total Observations'])
    missing_data.index.name ='Categorical Feature'
    st.write(missing_data)

    st.subheader("Missing Numerical Features")
    numeric_features = df.select_dtypes(include=[np.number])
    total = numeric_features.isnull().sum().sort_values(ascending=False)
    percent = ((numeric_features.isnull().sum()/numeric_features.isnull().count())*100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1,join='outer', keys=['Total Missing Count', '% of Total Observations'])
    missing_data.index.name =' Numeric Feature'
    st.write(missing_data)

if __name__ == "__main__":
    write()