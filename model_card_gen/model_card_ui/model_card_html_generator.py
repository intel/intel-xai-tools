from intel_ai_safety.model_card_gen.model_card_gen import ModelCardGen

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
    model_card_html = ModelCardGen.generate(model_card, metric_threshold_csv, metric_grp_csv)
    return model_card_html._repr_html_()
