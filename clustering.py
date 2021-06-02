import streamlit as st
import pandas as pd
from PIL import Image

def write():
    st.write(
        """
    # Clustering with KMeans Clustering

    """
    )
    st.write("Distortion of Different Number of Clusters")
    img = Image.open('clustering/distortion.png')
    st.image(img, use_column_width=True)

    st.write("KMeans Clustering with 3 Clusters")
    st.write("Scatter Plot of Real Clusters and KMeans Cluters")
    img = Image.open('clustering/cluster.png')
    st.image(img, use_column_width=True)

if __name__ == "__main__":
    write()
