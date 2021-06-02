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

df = pd.read_csv('Bank_CS.csv')
df.drop(columns=['Unnamed: 0','Unnamed: 0.1'], inplace=True)
## get numerical features
numeric_features = df.select_dtypes(include=[np.number])

year_feature = [feature for feature in numeric_features if 'Month' in feature or 'Year' in feature]
year_feature.remove('Monthly_Salary')

def write():
    with st.spinner("Loading EDA ..."):
        ast.shared.components.title_awesome("Exploratory Data Analysis")
    
    st.write(
        """
    # Temporal Features

    """
    )
    st.markdown('By visuaizing all the temporal features, some assumption and idea of the data can be grasp.')
    # for feature in year_feature:
    #     data=df.copy()
    #     g = sns.catplot(x='Decision', y=feature, kind="swarm", data=data, height=5, aspect=2)
    #     g.fig.suptitle(feature)
    #     st.pyplot(g)
    name = "Bank_EDA_cell_10_output_"
    for i in range(1,5):
        _ = name + str(i) + '.png'
        image = Image.open(_)
        st.image(image, use_column_width=True)

    st.subheader("Observations: ")
    st.markdown('---')
    st.markdown("**Is there any pattern found from plotting the temporal variable against the Desicion for loan application?**")
    st.markdown('For all the temporal features, the accepted case outnumbered the rejected cases.')

    st.markdown('---')
    st.markdown("**Does the duration of delayed credit carrd payment and the loan duration affect the loan application decision?**")
    st.markdown("From the plot of Credit Card Exceed Months and Loan Tenure Year against Decision, we can observe that in general, how long a person did not pay for his or her credit card and loan duration **_do not directly_** affect the result of the loan application.")

    st.markdown('---')
    st.markdown("**Is there any preference on year to financial freedom?**")
    st.markdown("From the plot, it seems like among all the rejected cases, cases with 14 and 15 years to financial freedom got rejected the most.")


if __name__ == "__main__":
    write()