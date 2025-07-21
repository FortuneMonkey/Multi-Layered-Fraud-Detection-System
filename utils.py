import tempfile
import os
import shap
import matplotlib.pyplot as plt

from fpdf import FPDF
from datetime import datetime

def generate_pdf_report(input_df, prediction_result, explanation_list=None, shap_plot=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Fraud Detection Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    # --- Transaction Details ---
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Transaction Details", ln=True)
    pdf.set_font("Arial", "", 12)
    for col in input_df.columns:
        val = input_df[col].values[0]
        pdf.cell(200, 8, txt=f"{col}: {val}", ln=True)

    pdf.ln(5)

    # --- Prediction Results ---
    pred_label = "Fraudulent" if prediction_result['prediction'] == 1 else "Legitimate"
    rec = "Flag for Manual Review" if prediction_result['prediction'] == 1 else "Proceed with Transaction"
    prob = round(prediction_result['probability'] * 100, 2)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Prediction Result", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 8, txt=f"Prediction: {pred_label}", ln=True)
    pdf.cell(200, 8, txt=f"Probability: {prob}%", ln=True)
    pdf.cell(200, 8, txt=f"Recommendation: {rec}", ln=True)

    # --- Base model predictions ---
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Base Model Probabilities", ln=True)
    pdf.set_font("Arial", "", 12)
    for model, p in prediction_result['model_probs'].items():
        pdf.cell(200, 8, txt=f"{model.upper()}: {round(p * 100, 2)}%", ln=True)

    # --- SHAP Explanation ---
    if explanation_list:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="SHAP Explanation", ln=True)
        pdf.set_font("Arial", "", 12)
        for line in explanation_list:
            pdf.multi_cell(0, 8, f"- {line}")

    # --- SHAP Plot as Image ---
    if shap_plot:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="SHAP Waterfall Plot", ln=True)
        pdf.image(shap_plot, x=10, w=190)

    # Save final PDF
    pdf_path = os.path.join(tempfile.gettempdir(), "fraud_prediction_report.pdf")
    pdf.output(pdf_path)
    return pdf_path


def explain_shap(raw_input_row, shap_values, feature_names, top_n=3):
    """
    Generate natural language SHAP explanations for top N impactful features.

    Args:
        raw_input_row (pd.Series): The original user input (unprocessed).
        shap_values (np.ndarray): SHAP values for the sample.
        feature_names (list): List of input feature names (after preprocessing).
        top_n (int): How many top contributing features to explain.

    Returns:
        List[str]: List of plain-English explanations.
    """
    explanation = []

    # --- Define Thresholds (Based on Your Dataset Summary) ---
    amount_threshold = 637
    young_age_threshold = 28
    new_account_threshold = 86
    odd_hour_low = 5
    odd_hour_high = 22
    large_quantity_threshold = 4

    # --- Get Top-N Contributing Features ---
    top_idx = abs(shap_values).argsort()[-top_n:][::-1]

    for i in top_idx:
        feat = feature_names[i]
        shap_val = shap_values[i]
        direction = "increased" if shap_val > 0 else "reduced"

        # Use original input for interpretation
        val = raw_input_row.get(feat, None)

        # Custom explanations
        if feat == "Transaction Amount":
            val = raw_input_row["Transaction Amount"]
            if val > amount_threshold:
                explanation.append(f"- High transaction amount (RM {val:.2f}) {direction} fraud risk.")
            else:
                explanation.append(f"- Normal transaction amount (RM {val:.2f}) {direction} fraud risk.")

        elif feat == "Customer Age":
            val = raw_input_row["Customer Age"]
            if val < young_age_threshold:
                explanation.append(f"- Young customer (age {int(val)}) {direction} fraud risk.")
            else:
                explanation.append(f"- Mature customer (age {int(val)}) {direction} fraud risk.")

        elif feat == "Account Age Days":
            val = max(0, int(raw_input_row["Account Age Days"]))
            if val < new_account_threshold:
                explanation.append(f"- New account (only {val} days old) {direction} fraud risk.")
            else:
                explanation.append(f"- Established account ({val} days) {direction} fraud risk.")

        elif feat == "Transaction Hour":
            val = int(raw_input_row["Transaction Hour"])
            if val < odd_hour_low or val > odd_hour_high:
                explanation.append(f"- Odd transaction time ({val}:00) {direction} fraud risk.")
            else:
                explanation.append(f"- Normal transaction time ({val}:00) {direction} fraud risk.")

        elif feat == "Quantity":
            val = raw_input_row["Quantity"]
            if val > large_quantity_threshold:
                explanation.append(f"- Large quantity purchased ({int(val)} items) {direction} fraud risk.")
            else:
                explanation.append(f"- Small quantity purchased ({int(val)} item{'s' if val > 1 else ''}) {direction} fraud risk.")

        elif feat.startswith("Device Used_"):
            device_type = feat.split("_")[1]
            if raw_input_row["Device Used"] == device_type:
                explanation.append(f"- Used a **{device_type}**, which {direction} fraud risk.")

        elif feat.startswith("Payment Method_"):
            method = feat.split("_")[1]
            if raw_input_row["Payment Method"] == method:
                explanation.append(f"- Payment made via **{method}**, which {direction} fraud risk.")

        elif feat.startswith("Product Category_"):
            category = feat.split("_")[1]
            if raw_input_row["Product Category"] == category:
                explanation.append(f"- Product category was **{category}**, which {direction} fraud risk.")

        elif feat == "Customer Location":
            loc = raw_input_row["Customer Location"]
            explanation.append(f"- Location: **{loc}**, which {direction} fraud risk.")

        else:
            # Default fallback
            val_disp = val if val is not None else "[unknown]"
            explanation.append(f"- Feature **{feat}** = {val_disp} {direction} fraud risk.")

    return explanation

