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
year_feature = [feature for feature in numeric_features if 'Month' in feature or 'Year' in feature]
year_feature.remove('Monthly_Salary')
## get discrete features
discrete_feature=[feature for feature in numeric_features if len(df[feature].unique())<25]
discrete_feature = list(set(discrete_feature) - set(year_feature))
#fig = plt.figure(figsize=(15, 10))
#ax = plt.subplots(nrows=3, ncols=3)

def write():
    with st.spinner("Loading EDA ..."):
        ast.shared.components.title_awesome("Exploratory Data Analysis")
    st.write(
        """
    # Discrete Features

    """
    )
    st.markdown("For discrete variables, a subplot is generated to observe and compare more easily.")
    fig, ax = plt.subplots(3,3, figsize=(15,10))
    c = 0
    for i in range(0, 3):
        for j in range(0, 3):
            if c < len(discrete_feature):
                sns.countplot(x = discrete_feature[c], hue = 'Decision', data=df, ax=ax[i][j])
                c+=1
            else:
                break
    fig.delaxes(ax[2][1])
    fig.delaxes(ax[2][2])
    fig.tight_layout()
    #fig.show()
    st.pyplot(fig)

    st.subheader("Observations: ")
    st.markdown('---')
    st.markdown('**Does the bank prefers providing loan for customers with less credit card in hand?**')
    st.markdown('Most loan applications came from customers with 4 credit cards on hand, and most of them are approved to get the loan.')

    st.markdown('---')
    st.markdown("**Do the score rating of a customer and number of loan to approve affect the loan application?**")
    st.markdown("The distribution of the accepted and rejected cases across the scores and number of loan applications are close to equal.")

    st.markdown('---')
    st.markdown("**Does how many properties a customer owns matters?**")
    st.markdown("Most customers own 2 properties, however for each number of properties group, the accepted cases are of the majority.")

    st.markdown('---')
    st.markdown("**How about number of dependents?**")
    st.markdown("Similarly, most customers fall in the category of having 2 dependents. For this group of customers, most of their applications got approved.")

    st.markdown('---')
    st.markdown("**How many bank accounts do most applicants own?**")
    st.markdown("Most applicants only own one, but the number of applicants slowly increases after a sharp drop in cases with 2 bank products, which brings both accepted and rejected cases to rise steadily")

    st.markdown('---')
    st.markdown('**Do customers with more side income get approval more easily?**')
    st.markdown("It can be observed that the number of approved applicants with 3 side incomes are slightly higher compared to lesser side incomes.")

if __name__ == "__main__":
    write()