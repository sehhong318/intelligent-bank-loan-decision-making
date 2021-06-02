import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import RepeatedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score

def write():
    st.write(
        """
    # Intelligent Decision-Making for Loan Application

    """
    )

    st.sidebar.header("User Input Parameters")


    def user_input_features():
        employement_type = st.sidebar.selectbox(
            "Employment Type",
            ("employer", "Self_Employed", "government", "employee", "Fresh_Graduate"),
        )
        more_than_one_products = st.sidebar.selectbox(
            "Using One or More products from the Bank", ("yes", "no")
        )
        credit_card_types = st.sidebar.selectbox(
            "Type of Credit Card possessed", ("platinum", "gold", "normal")
        )
        property_type = st.sidebar.selectbox("Type of Property possessed", ('condominium','bungalow','terrace','flat'))
        state = st.sidebar.selectbox(
            "State",
            (
                "Johor",
                "Selangor",
                "Kuala Lumpur",
                "Penang",
                "Negeri Sembilan",
                "Sarawak",
                "Sabah",
                "Terrenganu",
                "Kedah",
            ),
        )

        credit_card_exceed_months = st.sidebar.slider(
            "Months Exceeded for Credit Card payment", 1, 7, 6
        )
        loan_amount = st.sidebar.slider("Loan Amount", 100194.0, 799628.0, 500000.0)
        loan_tenure_year = st.sidebar.slider("Loan Tenure Years", 10, 24, 15)
        number_of_dependents = st.sidebar.slider("Number of dependents", 2, 6, 4)
        years_to_financial_freedom = st.sidebar.slider("Years to Financial Freedom", 5, 19, 4)
        number_of_credit_card_facility = st.sidebar.slider(
            "Number of credit card facility", 2, 6, 4
        )
        number_of_properties = st.sidebar.slider("Number of Properties owned:", 2, 5, 3)
        number_of_bank_products = st.sidebar.slider(
            "Number of Bank products owned:", 1, 5, 3
        )
        number_of_loan_to_Approve = st.sidebar.slider(
            "Number of Loans to be approved:", 1, 3, 2
        )
        years_for_properties_to_completion = st.sidebar.slider(
            "Years for properties to completion:", 10, 13, 12
        )
        number_of_side_income = st.sidebar.slider("Number of Side Income:", 1, 3, 2)
        monthly_salary = st.sidebar.slider("Monthly Salary:", 3583.0, 12562.0, 4000.0)
        total_sum_of_loan = st.sidebar.slider(
            "Total sum of loan:", 420239.0, 1449960.0, 600000.0
        )
        total_income_for_join_application = st.sidebar.slider(
            "Total Income for Joined Application:", 7523.0, 19995.0, 8000.0
        )
        score = st.sidebar.slider("Score:", 6, 9, 7)

        data = {
            "Credit_Card_Exceed_Months": credit_card_exceed_months,
            "Employment_Type": employement_type,
            "Loan_Amount": loan_amount,
            "Loan_Tenure_Year": loan_tenure_year,
            "More_Than_one_Products": more_than_one_products,
            "credit_card_types": credit_card_types,
            "Number_of_Dependents": number_of_dependents,
            "Years_to_financial_freedom": years_to_financial_freedom,
            "Number_of_Credit_Card_Facility": number_of_credit_card_facility,
            "Number_of_Properties": number_of_properties,
            "Number_of_bank_products": number_of_bank_products,
            "Number_of_Loan_to_Approve": number_of_loan_to_Approve,
            "Property_type": property_type,
            "Years_for_Property_to_Completion": years_for_properties_to_completion,
            "State": state,
            "Number_of_Side_Income": number_of_side_income,
            "Monthly_Salary": monthly_salary,
            "Total_Sum_of_Loan": total_sum_of_loan,
            "Total_Income_for_Join_Application": total_income_for_join_application,
            "Score": score,
        }
        features = pd.DataFrame(data, index=[0])
        features["Credit_Card_Exceed_Months"] = features[
            "Credit_Card_Exceed_Months"
        ].astype("int64")
        features["Employment_Type"] = features["Employment_Type"].astype("str")
        features["Loan_Amount"] = features["Loan_Amount"].astype("float64")
        features["Loan_Tenure_Year"] = features["Loan_Tenure_Year"].astype("float64")
        features["More_Than_one_Products"] = features["More_Than_one_Products"].astype(
            "str"
        )
        features["credit_card_types"] = features["credit_card_types"].astype("str")
        features["Number_of_Dependents"] = features["Number_of_Dependents"].astype("int64")
        features["Years_to_financial_freedom"] = features[
            "Years_to_financial_freedom"
        ].astype("float64")
        features["Number_of_Credit_Card_Facility"] = features[
            "Number_of_Credit_Card_Facility"
        ].astype("float64")
        features["Number_of_Properties"] = features["Number_of_Properties"].astype(
            "float64"
        )
        features["Number_of_bank_products"] = features["Number_of_bank_products"].astype(
            "float64"
        )
        features["Number_of_Loan_to_Approve"] = features[
            "Number_of_Loan_to_Approve"
        ].astype("float64")
        features["Property_type"] = features["Property_type"].astype("str")
        features["Years_for_Property_to_Completion"] = features[
            "Years_for_Property_to_Completion"
        ].astype("float64")
        features["State"] = features["State"].astype("str")
        features["Number_of_Side_Income"] = features["Number_of_Side_Income"].astype(
            "float64"
        )
        features["Monthly_Salary"] = features["Monthly_Salary"].astype("float64")
        features["Total_Sum_of_Loan"] = features["Total_Sum_of_Loan"].astype("float64")
        features["Total_Income_for_Join_Application"] = features[
            "Total_Income_for_Join_Application"
        ].astype("float64")
        features["Score"] = features["Score"].astype("int64")
        features.columns = features.columns.str.upper()
        return features


    df = user_input_features()

    st.subheader("User Input parameters")
    st.write(df)

    # processing the input data
    num_cols = df[["LOAN_AMOUNT","MONTHLY_SALARY","TOTAL_SUM_OF_LOAN","TOTAL_INCOME_FOR_JOIN_APPLICATION"]]
    cat_cols = df.drop(num_cols.columns, 1)
    for i in cat_cols:
        df[i] = df[i].astype(object)

    partial_dataset = pd.read_csv("partial_dataset.csv")
    partial_dataset.drop(columns=["Decision"], inplace=True)
    partial_dataset.columns = partial_dataset.columns.str.upper()
    temp_dataset_X = pd.concat([partial_dataset, df], ignore_index=True)

    num_cols = temp_dataset_X.select_dtypes(include=["int64", "float64"])

    for i in num_cols:
        temp_dataset_X[i] = pd.cut(
            temp_dataset_X[i], bins=3, precision=0, duplicates="drop"
        )

    temp_dataset_X[num_cols.columns] = temp_dataset_X[num_cols.columns].apply(
        LabelEncoder().fit_transform
    )
    temp_dataset_X[num_cols.columns] = temp_dataset_X[num_cols.columns].replace(
        {0: "Low", 1: "Medium", 2: "High"}
    )

    processed_input_data = temp_dataset_X.tail(1)
    processed_input_data.reset_index(drop=True, inplace=True)

    st.subheader("Processed User Input Parameters")
    st.write(processed_input_data.astype("object"))


    ##############################################################
    # encoding the input data

    # independents
    temp_dataset_X = pd.get_dummies(temp_dataset_X)
    processed_input_data = temp_dataset_X.tail(1)
    processed_input_data.reset_index(drop=True, inplace=True)

    #############################################################

    # selecting the columns
    input_dataset = processed_input_data.copy()
    input_dataset.columns = input_dataset.columns.str.upper()
    input_dataset.to_csv("input.csv", index=False)
    data_processed = pd.read_csv("dataset_processed.csv")
    input_dataset = input_dataset[data_processed.columns]

    ##############################################################

    # classification

    dtc_model = pickle.load(open("finalized_model_DecisionTreeClassifier.pkl", "rb"))
    result_dtc = dtc_model.predict(input_dataset)

    rfc_model = pickle.load(open("finalized_model_RandomForestClassifier.pkl", "rb"))
    result_rfc = rfc_model.predict(input_dataset)

    gbc_model = pickle.load(open("finalized_model_GradientBoostingClassifier.pkl", "rb"))
    result_gbc = gbc_model.predict(input_dataset)


    st.subheader("Prediction")
    decisions = np.array(["Reject", "Accept"])
    predicted_data = {
        "Decision Tree Classifier": decisions[result_dtc],
        "Random Forest Classifier": decisions[result_rfc],
        "Gradient Boosting Classifier": decisions[result_gbc],
    }
    prediction = pd.DataFrame(predicted_data, index=[0])

    st.write(prediction)


    st.subheader("Prediction Probability")
    st.subheader("Decision Tree Classifier")
    st.write(dtc_model.predict_proba(input_dataset))
    st.subheader("Random Forest Classifier")
    st.write(rfc_model.predict_proba(input_dataset))
    st.subheader("Gradient Boosting Classifier")
    st.write(gbc_model.predict_proba(input_dataset))


if __name__ == "__main__":
    write()