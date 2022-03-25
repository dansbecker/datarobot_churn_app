import altair as alt
import numpy as np
import streamlit as st
import datarobot as dr
import pandas as pd
from typing import Tuple, Dict

from project_metadata import dataset_id, deployment_id, target_name

alt.themes.enable("fivethirtyeight")

# @st.cache
def get_feature_impact(model: dr.Model) -> alt.Chart:
    feature_impacts = pd.DataFrame(model.get_or_request_feature_impact())
    # TODO: Better formatting of variable names
    # TODO: Limit number of rows
    out = (
        alt.Chart(feature_impacts, title="Relative Importance of Different Drivers")
        .mark_bar()
        .encode(
            alt.X("impactNormalized:Q", title=None),
            alt.Y("featureName", title=None, sort="-x"),
        )
    ).properties(width=800)
    return out


# @st.cache
def combine_lc_bins(lc: pd.DataFrame, n_result_bins: int = 10) -> pd.DataFrame:
    # TODO: Maybe handle cases where bin weights aren't consistent
    input_bins = lc["bin"].max() + 1  # bins are 0 indexed
    assert (
        input_bins % n_result_bins == 0
    ), "Tried to combine learning curve rows in invalid way"
    bin_size = input_bins / n_result_bins
    lc["Churn Risk Group"] = lc["bin"] // bin_size
    out = (
        lc.groupby(["Churn Risk Group", "variable"])
        .value.mean()
        .to_frame()
        .reset_index()
    )
    return out


@st.cache(allow_output_mutation=True)
def get_lift_chart(model: dr.Model):
    # TODO: be smarter about which lift chart to pull. Currently pulls first one listed
    # which is frequently validation. Sometimes holdout or cross val exist
    lc = (
        pd.DataFrame(model.get_all_lift_charts()[0].bins)
        .reset_index()
        .rename(columns={"index": "bin"})
        .melt(id_vars=["bin"], value_vars=["actual", "predicted"])
    )

    lc_to_plot = combine_lc_bins(lc)
    lc_to_plot["Percent Churning"] = lc_to_plot.value * 100

    out = (
        alt.Chart(lc_to_plot, title="Predicted vs Actual Churn By Risk Category")
        .mark_line()
        .encode(
            alt.X("Churn Risk Group", axis=alt.Axis(values=np.array(range(9)))),
            alt.Y("Percent Churning"),
            alt.Color("variable", title="Churn Measurement"),
        )
    ).properties(width=800)
    return out


@st.cache
def get_raw_data() -> pd.DataFrame:
    # TODO: FIX THIS FUNCTION.
    # It's reading full data basically as a workaround for https://datarobot.atlassian.net/browse/DSX-2141
    # Everything we do will be in sample, and this is hard to generalize to new projects
    dataset = dr.Dataset.get(dataset_id)
    dataset.get_file("temp.csv")
    return pd.read_csv("temp.csv")


@st.cache
def get_preds(data: pd.DataFrame) -> pd.DataFrame:
    # I considered model.request_training_predictions(dr.enums.DATA_SUBSET.HOLDOUT)
    # but it throws exception if called more than once & I don't see how to get prev results

    _, preds = dr.BatchPredictionJob.score_pandas(deployment_id, data)
    # We will create our own threshold and convert probabilities to labels with them. So drop these
    positive_class_name = str(preds.POSITIVE_CLASS[0])
    positive_pred_name = "_".join([target_name, positive_class_name, "PREDICTION"])
    out = preds.rename(columns={positive_pred_name: "predicted_churn_prob"})
    # TODO: Drop Negative class too
    unnecessary_cols = ["THRESHOLD", target_name + "_PREDICTION", "POSITIVE_CLASS"]
    out.drop(columns=unnecessary_cols, inplace=True)
    return out


@st.cache
def get_confusion_matrix(preds: pd.DataFrame, threshold: float) -> Dict[str, int]:
    preds_copy = preds.copy()
    preds_copy["take_action"] = preds.predicted_churn_prob > threshold
    confusion_matrix = {
        "True Positive": preds_copy.query("take_action & Churn").shape[0],
        "False Positive": preds_copy.query("take_action & (not Churn)").shape[0],
        "True Negative": preds_copy.query("(not take_action) & (not Churn)").shape[0],
        "False Negative": preds_copy.query("(not take_action) & Churn").shape[0],
    }
    return confusion_matrix
