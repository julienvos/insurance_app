from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model("deploy_some_model")


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df["Label"][0]

    return predictions


def run():

    from PIL import Image

    image = Image.open("picture.jfif")

    st.image(image, use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch")
    )

    st.sidebar.info("This app is to predict the insurance bill")
    st.sidebar.success("Some other text")

    st.title("The insurance app")

    if add_selectbox == "Online":

        age = st.number_input("Age", min_value=1, max_value=100, value=25)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.number_input("BMI", min_value=10, max_value=40, value=20)
        children = st.selectbox("How many children?", [1, 2, 3, 4, 5])
        if st.checkbox("Smoker"):
            smoker = "yes"
        else:
            smoker = "no"

        region = st.selectbox(
            "US Region", ["southwest", "northwest", "northeast", "southeast"]
        )

        output = ""

        input_dict = dict(
            age=age, sex=sex, bmi=bmi, children=children, smoker=smoker, region=region
        )
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model, input_df=input_df)
            output = "$" + str(output)

        st.success("The output is {}".format(output))

    if add_selectbox == "Batch":
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)


if __name__ == "__main__":
    run()
