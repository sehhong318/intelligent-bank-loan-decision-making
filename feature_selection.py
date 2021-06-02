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

boruta_score = pd.read_csv('boruta_score.csv')
rfe_score = pd.read_csv('rfe_score.csv')


def write():
    with st.spinner("Loading EDA ..."):
        ast.shared.components.title_awesome("Feature Selection")

    st.write(
        """
    # Feature Selection

    """
    )
    st.set_option('deprecation.showPyplotGlobalUse', False)
    ## Raw Data
    st.subheader("BORUTA")
    st.write(
        """
    #### Top 10 Features selected

    """
    )
    print('---------Top 10----------')
    st.write(boruta_score.head(10))
    st.write(
        """
    #### Bottom 10 Features selected

    """
    )
    st.write(boruta_score.tail(10))
    st.write("All features of Boruta")
    sns.catplot(x="Score", y="Features", data = boruta_score[:], kind = "bar", 
               height=14, aspect=1.9, palette='coolwarm')
    st.pyplot()

    top_features_boruta = boruta_score.loc[boruta_score['Score'] > 0.5,'Features'].values
    st.write('Total Number of Top Features based on Boruta')
    st.write(len(top_features_boruta))
    
    st.text("")
    st.text("")
    st.text("")


    st.subheader("RFE")
    st.write(
        """
    #### Top 10 Features selected

    """
    )
    print('---------Top 10----------')
    st.write(rfe_score.head(10))
    st.write(
        """
    #### Bottom 10 Features selected

    """
    )
    st.write(rfe_score.tail(10))
    st.write("All features of RFE")
    sns.catplot(x="Score", y="Features", data = rfe_score[:], kind = "bar", 
               height=14, aspect=1.9, palette='coolwarm')
    st.pyplot()

    top_features_rfe = rfe_score.loc[rfe_score['Score'] > 0.5,'Features'].values
    st.write('Total Number of Top Features based on Boruta')
    st.write(len(top_features_rfe))
    st.write()

    st.text("")
    st.text("")
    st.text("")

    st.write("Number of final features extracted by comparing both techniques")
    final_features = []
    for i in top_features_boruta:
        for j in top_features_rfe:
                if i==j:
                    final_features.append(i)

    st.write(len(final_features))
    st.write(final_features)

    st.subheader("Discussion")
    st.write("To extract an optimal set of features that will be applied in our predictive \
            models, features that have a score of 0.5 will be extracted from both techniques. Boruta holds 55 features as a set with a score of 0.5 and above while \
             RFE holds 76 features as a set with a score of 0.5 or higher. Both of these \
             features with the score of 0.5 are compared, and the features that existed in \
             both sets with the score of 0.5 and above are extracted to be an optimal set \
             of features. The number of features extracted to the optimal dataset is 51. \
             As both BORUTA and RFE use Random Forest Classification as a base for \
             performing feature selection, the optimal dataset changes slightly every time \
             the process is re-run. As the optimal set of features is being constructed, we then continue to sample our data based on the optimal set of features.")



if __name__ == "__main__":
    write()
