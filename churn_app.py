# To update for your data/model, make changes to the fields in project_metadata.py

import datarobot as dr
import streamlit as st

from project_metadata import dataset_id, deployment_id, target_name, dr_key
from understanding_churn_section import (
    get_confusion_matrix,
    get_feature_impact,
    get_lift_chart,
    get_preds,
    get_raw_data,
)
from anti_churn_section import uploaded_data_section

# st.write(get_lift_chart(model))
st.write("---")
st.header("Understanding Churn")
deployment = dr.Deployment.get(deployment_id)
model_id = deployment.model["id"]
project_id = deployment.model["project_id"]
project = dr.Project.get(project_id)
model = dr.Model.get(project_id, model_id)
st.write(get_feature_impact(model))
st.write(get_lift_chart(model))
st.write("---")
st.header("Targeting Your Anti-Churn Program")
st.write(
    """This section lets you choose a threshold churn risk for who you target with your anti-churn program.
Upload your file of new accounts and it will tell you who to target.
"""
)

raw_training_data = get_raw_data()
training_data_preds = get_preds(raw_training_data)


risk_threshold = (
    st.slider(
        "Threshold churn risk (in %) for taking churn prevention action", 0, 99, 1
    )
    / 100
)
confusion_matrix = get_confusion_matrix(training_data_preds, risk_threshold)
st.write(
    f"""If you used this risk threshold in the past, you would have:
- Contacted {confusion_matrix['True Positive']} of the churned accounts
- Not contacted {confusion_matrix['False Negative']} of the churned account
- Contacted {confusion_matrix['False Negative']} of the unchurned accounts
- Not contacted {confusion_matrix['True Negative']} of the unchurned accounts
"""
)

uploaded_file = st.file_uploader("Upload data to make a prediction", type="csv")
if uploaded_file is not None:
    uploaded_data_section(uploaded_file, risk_threshold)

st.write("---")
st.header("Outstanding TODOs")
st.write(
    """
- Let user select accounts for anti-churn effort based on multiple factors (including those in raw data). Not just predicted risk
- Show fraction with anti-churn measure in lift chart
- Let user specify uplift assumption and show economic results
"""
)
