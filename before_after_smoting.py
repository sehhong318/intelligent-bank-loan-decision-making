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



def write():
    with st.spinner("Loading EDA ..."):
        ast.shared.components.title_awesome("Sampling the data")

    st.write(
        """
    # Comparison of results before and after Sampling
    SMOTE-NC is used in this project for oversampling the minority class

    """
    )
    st.image(image1, caption='Results before applying SMOTE-NC')
    st.write("")
    st.image(image2, caption='Results after applying SMOTE-NC')
    st.write("")
    st.subheader("Findings")
    st.write("The results of the predictive models does not perform well on the dataset that has yet to be applied with SMOTE-NC. It is because that the dataset is imblanced with the \
        majority class as 'Accept' and minority class as 'Reject")






if __name__ == "__main__":
    write()
