import pandas as pd
import streamlit as st

st.set_option("deprecation.showfileUploaderEncoding", False)


def main():
    st.title("Web Log Analysis")
    uploaded_file = st.file_uploader("Upload a log file", type=["log"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if st.checkbox("Show summary statistics"):
            st.write(df.describe())

        if st.checkbox("Show column names"):
            st.write(df.columns)

        if st.checkbox("Show unique values in column"):
            column = st.selectbox("Select a column", df.columns)
            st.write(df[column].unique())


if __name__ == "__main__":
    main()
