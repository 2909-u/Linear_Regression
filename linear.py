import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# App title
st.title("Linear Regression App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load CSV
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(data)

    # Dropdowns for variable selection
    columns = data.columns
    x_column = st.selectbox("Select X (Independent Variable):", columns)
    y_column = st.selectbox("Select Y (Dependent Variable):", columns)

    if x_column and y_column:
        # Prepare data
        X = data[[x_column]].values  # X should be 2D
        y = data[y_column].values  # y can be 1D

        # Perform Linear Regression
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Regression parameters
        coef = model.coef_[0]
        intercept = model.intercept_
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        # Display results
        st.subheader("Regression Results")
        st.write(f"Equation of the line: **y = {coef:.2f}x + {intercept:.2f}**")
        st.write(f"R-squared: **{r2:.2f}**")
        st.write(f"Mean Squared Error: **{mse:.2f}**")

        # Plotting
        fig, ax = plt.subplots()
        ax.scatter(X, y, color="blue", label="Actual Data")
        ax.plot(X, y_pred, color="red", label="Regression Line")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.legend()
        st.pyplot(fig)
