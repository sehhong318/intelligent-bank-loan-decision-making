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
categorical_features = df.select_dtypes(include=[np.object])

def write():
    with st.spinner("Loading EDA ..."):
        ast.shared.components.title_awesome("Exploratory Data Analysis")
    
    st.write(
        """
    # Categorical Features

    """
    )
    st.markdown("Count Plots visualizes the distibution of categorical variables.")
    cat_cols = categorical_features.columns.tolist()
    fig, ax =plt.subplots(3,2, figsize=(15,16))
    c = 0
    for i in range(0, 3):
        for j in range(0, 2):
            if c < len(categorical_features):
                g = sns.countplot(x = cat_cols[c], hue = 'Decision', data=df, ax=ax[i][j])
                g.set_xticklabels(g.get_xticklabels(), rotation=45)
                c+=1
            else:
                break
    fig.tight_layout()
    #fig.show()
    st.pyplot(fig)

    st.subheader("Observations: ")
    st.markdown('---')
    st.markdown('**Customers from what type of employment apply for loan the least?**')
    st.markdown("In this dataset, customers that are self employed are of the minority group, and employment type does not really affect the decision.")

    st.markdown('---')
    st.markdown("**How many accouts are preferred when applying for loan?**")
    st.markdown("The number of accounts does not affect the decision. Both counts are similar in terms of accepted and also rejected cases.")

    st.markdown('---')
    st.markdown("**Does platinum credit card customers are in least demand of getting a loan?**")
    st.markdown("Yes. Most customers have normal credit cards and there are least of them having platinum credit cards.")

    st.markdown('---')
    st.markdown("**What is the most common property that customers applying loan for?**")
    st.markdown("Condominium.")

    st.markdown('---')
    st.markdown("**Where do majority loan applications come from?**")
    st.markdown("Most of the applications come from customers from Kuala Lumpur, with Johor being the runner up.")

    st.markdown('---')
    st.markdown("**What can we conclude from the count of Decision?**")
    st.markdown("The dataset is **imbalanced**, with the number of accepted cases being **_a lot greater_** than the number of rejected cases.")

    st.markdown('---')
    st.markdown("**What measure should be taken from the conclusion drawn above?**")
    st.markdown("Data imbalance treatment such as SMOTE-NC needs to be performed.")

    
if __name__ == "__main__":
    write()
