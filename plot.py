import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# --- Plotly: Feature Importance Bar Plot ---
def plot_feature_importance(df, title="Feature Importance"):
    fig = px.bar(
        df.sort_values(by="importance", ascending=False),
        x="importance",
        y="feature",
        orientation="h",
        title=title,
        height=450
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

# --- Plotly: SHAP Importance Bar Plot ---
def plot_shap_importance(df, title="SHAP Global Summary"):
    fig = px.bar(
        df.sort_values(by="Mean |SHAP|", ascending=False),
        x="Mean |SHAP|",
        y="Feature",
        orientation="h",
        title=title,
        height=450
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

# --- Plotly: Interactive Confusion Matrix ---
def create_interactive_confusion_matrix(cm_data, labels, title="Confusion Matrix"):
    """
    Creates an interactive Plotly confusion matrix.

    Args:
        cm_data (list or np.array): The confusion matrix data, e.g., [[TN, FP], [FN, TP]].
        labels (list): The class labels, e.g., ['Legitimate', 'Fraudulent'].
        title (str): The title for the chart.

    Returns:
        go.Figure: A Plotly figure object.
    """
    # Ensure data is a numpy array for calculations
    cm_data = np.array(cm_data)

    fig = go.Figure(data=go.Heatmap(
        z=cm_data,
        x=labels,
        y=labels,
        colorscale='Blues',
        hoverongaps=False)
    )

    # Add annotations (the numbers in the cells)
    annotations = []
    for i, row in enumerate(cm_data):
        for j, value in enumerate(row):
            annotations.append(
                go.layout.Annotation(
                    text=f"{value:,}", # Format number with commas
                    x=labels[j],
                    y=labels[i],
                    xref="x1",
                    yref="y1",
                    showarrow=False,
                    # Set font color based on cell background
                    font=dict(color="white" if value > cm_data.max() / 1.5 else "black")
                )
            )

    fig.update_layout(
        width=600,  
        height=500, 
        title_text=f'<b>{title}</b>',
        xaxis_title='<b>Predicted Label</b>',
        yaxis_title='<b>True Label</b>',
        annotations=annotations,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed'), # Puts the first label at the top
        font=dict(size=14)
    )

    return fig


# Simulated feature importance 
feature_importance_lgbm = pd.DataFrame({
    'feature': [
        "Quantity", "Customer_Age", "Account_Age_Days", "Transaction_Amount",
        "Customer_Location", "Transaction_Hour", "Transaction_Day",
        "Product_Category_electronics", "Product_Category_toys_&_games", "Product_Category_home_&_garden"
    ],
    'importance': [1942, 1581, 1474, 1405, 1329, 1041, 783, 491, 484, 475]
})

feature_importance_xgb = pd.DataFrame({
    'feature': [
        "Account_Age_Days", "Quantity", "Customer_Location", "Transaction_Amount",
        "Transaction_Hour", "Customer_Age", "Transaction_Day",
        "Device_Used_tablet", "Payment_Method_bank_transfer", "Payment_Method_credit_card"
    ],
    'importance': [875, 674, 664, 616, 561, 546, 362, 257, 237, 233]
})

feature_importance_rf = pd.DataFrame({
    'feature': [
        "Transaction_Hour", "Account_Age_Days", "Device_Used_tablet", "Quantity",
        "Device_Used_desktop", "Device_Used_mobile", "Payment_Method_credit_card",
        "Transaction_Amount", "Payment_Method_PayPal", "Payment_Method_debit_card"
    ],
    'importance': [0.092015, 0.091503, 0.087045, 0.083245, 0.082502, 0.078758, 0.055459, 0.050243, 0.050143, 0.049721]
})

# Simulated SHAP values (mean absolute values)
shap_importance_lgbm = pd.DataFrame({
    'Feature': [
        "Quantity", "Device Used_desktop", "Device Used_mobile",
        "Payment Method_credit card", "Payment Method_bank transfer", "Payment Method_debit card",
        "Device Used_tablet", "Payment Method_PayPal", "Product Category_home & garden", "Product Category_electronics"
    ],
    'Mean |SHAP|': [1.609626, 1.199493, 1.120737, 1.116978, 1.089387, 1.066800, 1.056574, 1.055769, 1.026705, 0.994231]
})

shap_importance_xgb = pd.DataFrame({
    'Feature': [
        "Quantity", "Device Used_desktop", "Device Used_tablet",
        "Device Used_mobile", "Payment Method_debit card", "Payment Method_PayPal",
        "Payment Method_bank transfer", "Payment Method_credit card", "Product Category_clothing", "Product Category_electronics"
    ],
    'Mean |SHAP|': [1.648680, 1.107018, 1.088141, 1.083608, 1.078347, 1.071654, 1.058340, 1.034990, 0.960198, 0.956538]
})

shap_importance_rf = pd.DataFrame({
    'Feature': [
        "Quantity", "Device Used_desktop", "Device Used_mobile",
        "Payment Method_credit card", "Payment Method_bank transfer", "Payment Method_debit card",
        "Device Used_tablet", "Payment Method_PayPal", "Product Category_home & garden", "Product Category_electronics"
    ],
    'Mean |SHAP|': [1.609626, 1.199493, 1.120737, 1.116978, 1.089387, 1.066800, 1.056574, 1.055769, 1.026705, 0.994231]
})

shap_importance_logreg = pd.DataFrame({
    'Feature': [
        "LightGBM", "XGBoost", "Random Forest"],

    'Mean |SHAP|': [1.250879, 1.222061, 1.029828]
})

