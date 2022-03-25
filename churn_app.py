import datarobot as dr
import streamlit as st

from project_metadata import dataset_id, deployment_id, target_name, dr_key
from train_data_utils import (
    get_confusion_matrix,
    get_feature_impact,
    get_lift_chart,
    get_preds,
    get_raw_data,
)
from new_data_utils import uploaded_data_section

deployment = dr.Deployment.get(deployment_id)
model_id = deployment.model["id"]
project_id = deployment.model["project_id"]
project = dr.Project.get(project_id)
model = dr.Model.get(project_id, model_id)
st.write(get_feature_impact(model))
risk_threshold = (
    st.slider(
        "Threshold churn risk (in %) for taking churn prevention action", 0, 99, 1
    )
    / 100
)
st.write(get_lift_chart(model))

st.write("Create confusion based on threshold")

uploaded_file = st.file_uploader("Upload data to make a prediction", type="csv")
if uploaded_file is not None:
    uploaded_data_section(uploaded_file, risk_threshold)

st.write(
    "TODOS: Let user select whether to take action based on multiple factors (including those in raw data). Not just predicted risk"
)
st.write("Show confusion matrix data")
st.write("Show fraction with anti-churn measure in lift chart")
st.write("Let user specify uplift assumption and show results")
