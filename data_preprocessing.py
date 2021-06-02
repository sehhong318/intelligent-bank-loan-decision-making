import warnings
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
from PIL import Image



image1 = Image.open('before_smoting.png')
image2 = Image.open('after_smoting.png')
df = pd.read_csv('Bank_CS.csv')
df_cleaned = pd.read_csv('cleaned_dataset.csv')
df_cleaned_encoded = pd.read_csv('cleaned_dataset_encoded.csv')


def write():
    with st.spinner("Loading ..."):
        ast.shared.components.title_awesome("Data Pre-processing")

    df = pd.read_csv('Bank_CS.csv')

    st.write(
        """
    # Data Pre-processing
    This section provides the steps of data pre-processing.

    """
    )
    st.header('Examples of Raw Data')
    st.write(df.head())
    st.write("")

    st.write('The size of our Raw Data')
    st.write(df.shape)
    st.write("")

    st.header("Dropping of irrelevant columns")
    
    st.write(df.shape)
    df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], inplace=True)
    st.write(df.head())
    st.write('Two irrelavent columns were dropped.')
    st.write("")

    st.header("Dealing with missing values, inconsistent data in a columns and binning nemrical data")
    st.write(
        """
    Firstly, missing values are filled using median of numerical data such as 'Loan_Amount','Monthly_Salary', 'Total_Sum_of_Loan', 'Total_Income_for_Join_Application'.
    Then, we filled in the categorical missing data using the most count of the data for each column.
    We futher binned 'Loan_Amount','Monthly_Salary', 'Total_Sum_of_Loan', 'Total_Income_for_Join_Application' in 'Low','Medium' and 'High' to ease our modelling process.
    Below shows the sample dataset after the processes have been done.

    """
    )
    st.write(df_cleaned.head())
    st.write("")
    st.header(
        "Encoding the data")
    st.write(
        """
    We have encoded the independant variables using dummification and the dependant variable by changing the "Reject" and "Accept" with 0 and 1.
    The result of encoding is shown below.

    """
    )
    st.write(df_cleaned_encoded.head())
    
    st.write("")
    st.write("We then proceeded with Feature selection, then sampling of the data.")


if __name__ == "__main__":
    write()
