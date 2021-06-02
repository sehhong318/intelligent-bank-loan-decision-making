import streamlit as st
import pandas as pd

df = pd.read_csv('arm_head.csv')

def write():
    st.write(
        """
    # Association Rule Mining

    """
    )
    st.write(df)

if __name__ == "__main__":
    write()