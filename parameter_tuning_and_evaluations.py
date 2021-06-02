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

accuracy_dtc = Image.open('accuracy dtc.png')
accuracy_gb = Image.open('accuracy gradient boosting.png')
accuracy_rfc = Image.open('accuracy rfc.png')
n_est_gb = Image.open('n_est gb.png')
n_est_rf = Image.open('n_est random forest.png')
max_depth_dt = Image.open('decision tree max depth.png')
max_depth_rf = Image.open('max depth random forest.png')
final_result = Image.open('final result.png')


def write():
    with st.spinner("Loading EDA ..."):
        ast.shared.components.title_awesome("Parameter Tuning and Model Evaluation")

    st.write(
        """
    # Parameter Tuning and Model Evaluation

    """
    )
    st.subheader('Parameter Tuning')
    st.subheader('Decision Tree Classifier')
    st.image(max_depth_dt,
             caption='AUC score of train and test set against the number of max_depth')
    st.write("")
    st.write("We can clearly see that when the AUC score of \
             training set continues to increase and the AUC score of test set started \
             to incline when the number of max depth approaches 5. We selected 3 as \
             the max depth for the Decision Tree Classifier as both of them holds the \
             score that is above 0.80 and both of them does not differ significantly. The \
             same process is being repeated on the n estimators and the max depth of \
             the Random Forest Classifier and the n estimators for Gradient Boosting \
             Classification.")
    st.subheader('Random Tree Classifier')
    st.image(n_est_rf,
             caption='AUC score of train and test set against the number of estimators')
    st.write("")
    st.write("Both of the AUC score reaches its peak in betweenthe number of 25 n estimators to 50. A number of 30 n estimators has been \
             picked to be included as the parameter")
    st.image(max_depth_rf,
             caption='AUC score of train and test set against the number of max_depth')
    st.write("")
    st.write("The AUC score for both the train set and the test \
             set remains as flat when the number of max depth reaches 5 where the \
             gap between both scores starts to lengthen as the number of max depth \
             increases. A number of 5 has been used as the number of max depths for \
             the parameter in Random Forest Classification.")
    st.subheader('Gradient Boosting Classifier')
    st.image(n_est_gb,
             caption='AUC score of train and test set against the number of estimators')
    st.write("")
    st.write("AUC score of test set remains straight at the score \
             of 0.80 while the AUC score of training set increases and reaches the peak at \
             1.0 at when the number of estimators hits 75. 100 has been selected as the \
             number of estimators in the parameter of Gradient Boosting Classification")

    st.subheader('Accuracy based on the numbers of optimal features used')
    st.subheader('Decision Tree Classifier')
    st.image(accuracy_dtc,
             caption='Accuracy against the number of optimal features used')
    st.write("")
    st.subheader('Random Tree Classifier')
    st.image(accuracy_rfc,
             caption='Accuracy against the number of optimal features used')
    st.write("")
    st.subheader('Gradient Boosting Classifier')
    st.image(accuracy_gb,
             caption='Accuracy against the number of optimal features used')
    st.write("")
    st.write("")

    st.subheader('Evaluation of Results')
    st.image(final_result)
    st.write("")
    st.write("The results show that out of all three classification \
             models implemented, the Random Forest Classification model delivers the \
             highest AUC score. Even though the accuracy of the Gradient Boosting \
             Classifier is good, the discrepancy between the accuracy on the training set \
             and the test set tells us that the model is slightly over fitted. It also has a \
             higher rate of False Negatives than other applied classification models. Random forest classifier holds the highest precision score and F1 score relative \
             to the other models used in classification.")


if __name__ == "__main__":
    write()
