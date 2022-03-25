from typing import Dict
import datarobot as dr
import pandas as pd
import streamlit as st
from project_metadata import deployment_id, target_name, account_id_name


def uploaded_data_section(
    uploaded_file: st.uploaded_file_manager.UploadedFile, risk_threshold: float
):
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        # st.write("Top of Uploaded Data")
        # st.write(uploaded_df.head())
    except:
        st.write("Unable to load uploaded data as CSV")
        st.write("TODO: Improve error handling for these files")
    try:
        preds = get_datarobot_predictions(uploaded_df)
    except:
        st.write(
            "Failed to get predictions. Check your uploaded data has correct columns"
        )
        st.write("TODO: Improve error handling here")
    account_action_list = choose_accounts_for_action(preds, risk_threshold)
    accounts_with_action = account_action_list.take_action.sum()
    accounts_without_action = account_action_list.shape[0] - accounts_with_action
    st.write(
        f"With current threshold you will take anti-churn action on {accounts_with_action} accounts."
    )
    st.write(f"You will take no action on {accounts_without_action} accounts.")
    add_results_download_button(account_action_list)


def add_results_download_button(results: pd.DataFrame):
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download list of accounst to take action with",
        csv,
        "churn_action_list.csv",
        "text/csv",
        key="download-csv",
    )


@st.cache
def get_datarobot_predictions(pred_data: pd.DataFrame) -> pd.DataFrame:
    job, preds = dr.BatchPredictionJob.score_pandas(deployment_id, pred_data)
    # Could assert preds.POSITIVE_CLASS.nunique() == 1
    pos_class_name = preds.POSITIVE_CLASS.iloc[0]
    pred_prob_name = "_".join([target_name, str(pos_class_name), "PREDICTION"])
    preds.rename(columns={pred_prob_name: "prediction"}, inplace=True)
    cols_to_keep = list(pred_data.columns) + ["prediction"]
    out = preds[cols_to_keep]
    return out


@st.cache
def choose_accounts_for_action(preds: pd.DataFrame, risk_threshold: float):
    out = preds.copy(deep=True)
    out["take_action"] = out.prediction > risk_threshold
    return out[[account_id_name, "take_action"]]
