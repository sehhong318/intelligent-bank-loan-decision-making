"""Main module for the streamlit app"""
import streamlit as st
import awesome_streamlit as ast
import eda
import temporal
import discrete
import continuous
import categorical
import missing
import correlation
import ml
import data_preprocessing
import home
import arm
import feature_selection
import before_after_smoting
import clustering
import parameter_tuning_and_evaluations

ast.core.services.other.set_logging_format()

PAGES = {
    "Home": home,
    "Exploratory Data Analysis": eda,
    "Temporal Variables Analysis": temporal,
    "Discrete Variables Analysis": discrete,
    "Continuous Variables Analysis": continuous,
    "Categorical Variables Analysis": categorical,
    "Missing Values Analysis": missing,
    "Correlation Analysis": correlation,
    "Data Pre-processing": data_preprocessing,
    "Association Rule Mining": arm,
    "Clustering": clustering,
    "Feature Selection": feature_selection,
    "Comparison before and after Sampling": before_after_smoting,
    "Parameter Tuning and Evaluations": parameter_tuning_and_evaluations,
    "Prediction Models": ml
}

def main():
    """Main function of the App"""
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)

if __name__ == "__main__":
    main()
