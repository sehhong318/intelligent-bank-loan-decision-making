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
## get numerical features
numeric_features = df.select_dtypes(include=[np.number])

def write():
    with st.spinner("Loading EDA ..."):
        ast.shared.components.title_awesome("Exploratory Data Analysis")

    st.write(
        """
    # Exploratory Data Analysis

    """
    )

    ## Raw Data
    st.markdown('The Exploratory data analysis is performed on raw data to study and understand the data before proceedinng to any machine learning work.')
    st.subheader("Raw Data")
    st.write(df)

    ## Describe the data
    st.markdown('The first thing to do is to observe the basic statistical information about the data.')
    st.subheader("Description of Data")
    st.write(df.describe().transpose())

if __name__ == "__main__":
    write()