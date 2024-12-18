import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO

# App title
st.title("Advanced Linear Regression App")

# Sidebar for file upload
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Data cleaning
    if data.isnull().values.any():
        st.warning("The dataset contains missing values.")
        if st.checkbox("Drop rows with missing values"):
            data = data.dropna()
        elif st.checkbox("Fill missing values with column mean"):
            data = data.fillna(data.mean())

    # Dataset statistics
    st.subheader("Dataset Statistics")
    st.write(data.describe())

    # Correlation heatmap
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Variable selection
    st.sidebar.subheader("Select Variables")
    columns = data.columns
    x_columns = st.sidebar.multiselect("Select Independent Variable(s):", columns)
    y_column = st.sidebar.selectbox("Select Dependent Variable:", columns)

    if x_columns and y_column:
        # Prepare data
        X = data[x_columns].values
        y = data[y_column].values

        # Perform Linear Regression
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Regression results
        coef = model.coef_
        intercept = model.intercept_
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        st.subheader("Regression Results")
        st.write(f"Intercept: **{intercept:.2f}**")
        st.write("Coefficients:")
        for i, col in enumerate(x_columns):
            st.write(f"- **{col}**: {coef[i]:.2f}")
        st.write(f"R-squared: **{r2:.2f}**")
        st.write(f"Mean Squared Error: **{mse:.2f}**")

        # Plotting options
        st.sidebar.subheader("Customize Plot")
        scatter_color = st.sidebar.color_picker("Scatter Plot Color", "#1f77b4")
        line_color = st.sidebar.color_picker("Regression Line Color", "#ff7f0e")
        marker_style = st.sidebar.selectbox("Marker Style", ["o", "s", "D", "^"])
        line_style = st.sidebar.selectbox("Line Style", ["-", "--", "-.", ":"])

        # Scatter plot and regression line
        st.subheader("Regression Plot")
        fig, ax = plt.subplots()
        if X.shape[1] == 1:  # Single regression plot
            ax.scatter(X, y, color=scatter_color, label="Data", marker=marker_style)
            ax.plot(X, y_pred, color=line_color, linestyle=line_style, label="Regression Line")
        else:  # Multiple regression pair plots
            for i, col in enumerate(x_columns):
                ax.scatter(X[:, i], y, label=f"{col} vs {y_column}", alpha=0.7)
        ax.set_xlabel(", ".join(x_columns))
        ax.set_ylabel(y_column)
        ax.legend()
        st.pyplot(fig)

        # Export results
        results = pd.DataFrame({"Actual": y, "Predicted": y_pred})
        csv = BytesIO()
        results.to_csv(csv, index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv.getvalue(),
            file_name="regression_results.csv",
            mime="text/csv",
        )

        # Prediction
        st.subheader("Make Predictions")
        new_data = {}
        for col in x_columns:
            new_data[col] = st.number_input(f"Enter value for {col}:", value=0.0)
        if st.button("Predict"):
            new_values = np.array([list(new_data.values())])
            prediction = model.predict(new_values)[0]
            st.write(f"Predicted value for {y_column}: **{prediction:.2f}**")
