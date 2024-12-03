from intel_ai_safety.model_card_gen.model_card_gen import ModelCardGen
import streamlit as st


def generate_error_message(column_name, metric_file_type):
    help_message = (
        "Please [click here](https://github.com/intel/intel-xai-tools/tree/main/model_card_gen/intel_ai_safety/model_card_gen/docs/examples/csv) "
        "to view examples of metric files. To learn how to create these files or see a step-by-step example, you can follow this "
        "[link](https://github.com/intel/intel-xai-tools/blob/main/notebooks/model_card_gen/hugging_face_model_card/hugging-face-model-card.ipynb) "
        "for further guidance."
    )
    if metric_file_type:
        help_message = (
            f"Error: The uploaded CSV file for {metric_file_type} does not contain the required column '{column_name}'. "
            + help_message
        )
    return help_message


def handle_exception(error_str):
    if "No column named" in error_str:
        column_name = error_str.split("'")[1]
        metric_file_type = (
            "Metrics by Threshold"
            if column_name in ["threshold", "accuracy", "precision", "recall", "f1"]
            else "Metrics by Group"
        )
        st.error(generate_error_message(column_name, metric_file_type))
    elif "range() arg 3 must not be zero" in error_str:
        st.error(
            "Error: The uploaded CSV file for Metrics by Group contains an empty 'group' column. "
            + generate_error_message("", "Metrics by Group")
        )
    elif "'Overall'" in error_str:
        st.error(
            "Error: Please check if the 'Overall' value exists in the 'feature' and 'group' columns in the uploaded Metric CSV files. "
            + generate_error_message("", "")
        )
    else:
        st.error(generate_error_message("", ""))


def generate_model_card_html(model_card, metric_threshold_csv=None, metric_grp_csv=None):
    """
    Generates an HTML representation of a model card, optionally including data from CSV files
    for metric thresholds and metric groups.

    Parameters:
    model_card (ModelCard): The model card object containing the model's metadata and other details.
    metric_threshold_csv (str, optional): The file path to a CSV containing metric threshold data.
    metric_grp_csv (str, optional): The file path to a CSV containing metric group data.

    Returns:
    str: The HTML representation of the model card.
    """
    try:
        model_card_html = ModelCardGen.generate(model_card, metric_threshold_csv, metric_grp_csv)
        return model_card_html._repr_html_()
    except Exception as e:
        handle_exception(str(e))
