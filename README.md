# Churn Reduction App

This is an early version of the anti-churn app. **It still requires iteration to become a valuable app. I (Dan Becker) am open to suggestions, questions, complaints and PRs so make those improvements.**

The goal is to let anyone put their model_id, dataset_id and similar metadata in `metadata.py` to use this app with their data/model.

Run the app locally with `streamlit run churn_app.py`

The app lets you upload a dataset to make recommendations for your anti-churn program. The data you upload must have the same format as the training data for your model. If you want to test the app with the current model, you can find test data to upload in `sample_data.csv`.

Some Changes I'm Considering:

- Let user select accounts for anti-churn effort based on multiple factors (including those in raw data). Not just predicted risk
- Let user specify uplift assumption and show economic results

[Demo of the app](https://share.streamlit.io/dansbecker/datarobot_churn_app/main/churn_app.py)
