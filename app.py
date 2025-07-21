# app.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import numpy as np
import tempfile
import os
import hashlib
import plotly.express as px
import streamlit.components.v1 as components
import plotly.graph_objects as go

from utils import generate_pdf_report
from datetime import datetime
from predict import predict_fraud, get_shap_input
from streamlit_extras.metric_cards import style_metric_cards
from utils import explain_shap
from streamlit_option_menu import option_menu

from plot import plot_feature_importance, plot_shap_importance,create_interactive_confusion_matrix
from plot import feature_importance_lgbm, feature_importance_xgb, feature_importance_rf, shap_importance_logreg

# --- Page Configuration ---
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# --- Sidebar Navigation ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 300px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    selected_page = option_menu(
        menu_title="Navigation",      
        options=["Single Transaction", "Batch Analysis", "Transaction History", "About"],  
        icons=["credit-card-2-front", "files", "clock","house"],  
        menu_icon="cast",             
        default_index=0,           
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "#262730", 
                "border-radius": "5px"
            },
            "icon": {
                "color": "#ff6347", 
                "font-size": "20px"
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#e0e0e0", 
                "color": "#FFFFFF" 
            },
            "nav-link-selected": {
                "background-color": "#28a745", 
                "color": "white"
            },
            "menu-title": {
                "font-size": "22px",
                "font-weight": "bold",
                "color": "#FFFFFF"
                
            }
        }
    )

# --- PAGE 1: Single Transaction Analysis ---

if selected_page == "Single Transaction":

    # --- Header ---
    st.markdown("<h1 style='text-align: center;'>🛡️ Fraud Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: grey;'>Powered by Ensemble Machine Learning</h5>", unsafe_allow_html=True)
    st.markdown("---")

    st.title("💳 Single Transaction Analysis")


    # --- Layout: Input Form ---
    with st.container():
        st.subheader("📋 Enter Transaction Details")
        with st.form("fraud_form"):
            col1, col2 = st.columns(2)

            with col1:
                transaction_amount = st.number_input("💰 Transaction Amount (RM)", min_value=0.0, step=0.1, help="Total value of the transaction in Malaysian Ringgit (RM).")
                quantity = st.number_input("📦 Quantity", min_value=1, step=1, help="Number of items in this transaction.")
                customer_age = st.number_input("🧑 Customer Age", min_value=1, max_value=100, help="Age of the customer making the transaction.")
                account_age = st.number_input("📅 Account Age (Days)", min_value=1, help="How many days the customer account has been active.")
                transaction_date = st.date_input("🗓️ Transaction Date", value=datetime.today(), help="Date when the transaction occurred.")

            with col2:
                payment_method = st.selectbox("💳 Payment Method", ["credit card", "debit card", "PayPal", "bank transfer"], help="Method used to pay for the transaction.")
                product_category = st.selectbox("🛍️ Product Category", ["electronics", "clothing", "home & garden", "health & beauty", "toys & games"], help="Category of product(s) purchased in this transaction.")
                device_used = st.selectbox("📱 Device Used", ["desktop", "mobile", "tablet"], help="Type of device used to perform the transaction.")
                customer_location = st.text_input("📍 Customer Location", value="Malaysia", help="Customer's city, region, or country (used for location-based fraud patterns).")
                transaction_hour = st.slider("⏰ Transaction Hour", 0, 23, 14, help="Hour of the day when the transaction occurred (0 = midnight, 23 = 11PM).")

            submit_btn = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)

    # Run Prediction if form is submitted
    if submit_btn:
        with st.spinner("🔍 Analyzing Transaction..."):
            raw_input = pd.DataFrame([{
                "Transaction Amount": transaction_amount,
                "Quantity": quantity,
                "Customer Age": customer_age,
                "Account Age Days": account_age,
                "Transaction Hour": transaction_hour,
                "Transaction Date": pd.to_datetime(transaction_date),
                "Payment Method": payment_method,
                "Product Category": product_category,
                "Device Used": device_used,
                "Customer Location": customer_location
            }])

            result = predict_fraud(raw_input)

            # Store in session state for download
            st.session_state['raw_input'] = raw_input
            st.session_state['result'] = result


    # --- Display Results ---
    if 'raw_input' in st.session_state and 'result' in st.session_state:
        raw_input = st.session_state['raw_input']
        result = st.session_state['result']

        st.markdown("---")
        st.subheader("🎯 Prediction Results")

        colA, colB, colC = st.columns(3)
        label = "🟢 Legitimate" if result['prediction'] == 0 else "🔴 Fraudulent"
        recommendation = "✅ Proceed with Transaction" if result['prediction'] == 0 else "⚠️ Flag for Manual Review"

        z1, z2, z3 = st.columns(3)
        z1.metric(label="Prediction", value=label, delta=None, delta_color="normal")
        z2.metric(label="Probability", value=f"{result['probability']:.2%}", delta=None, delta_color="normal")
        z3.metric(label="Recommendation", value=recommendation, delta=None, delta_color="normal")
        style_metric_cards(background_color="#262730", border_left_color="#1f77b4", border_size_px=1)
        st.markdown("---")

        
        result_df = raw_input.copy()
        result_df["Fraud_Prediction"] = ["Fraudulent" if result['prediction'] == 1 else "Legitimate"]
        result_df["Fraud_Probability"] = [round(result["probability"], 4)]

    
        # --- Downloadable CSV Report ---
        result_df["Fraud_Probability"] = (result_df["Fraud_Probability"] * 100).round(2).astype(str) + '%'
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prediction Report (CSV)",
            data=csv,
            file_name="fraud_prediction_report.csv",
            mime='text/csv'
        )

        # --- Downloadable PDF Report ---
        # --- SHAP Setup ---
        model, shap_input = get_shap_input(raw_input)
        explainer = shap.Explainer(model)
        shap_values = explainer(shap_input)

        # --- Generate text explanation
        shap_list = explain_shap(
            raw_input.iloc[0], 
            shap_values[0].values, 
            shap_input.columns
        )

        # --- Generate and save SHAP plot as image

        shap.plots.waterfall(shap_values[0], max_display=10)
        shap_plot_path = os.path.join(tempfile.gettempdir(), "shap_plot.png")
        plt.savefig(shap_plot_path, bbox_inches="tight")
        plt.close()

        # --- Generate PDF report with SHAP plot image
        pdf_path = generate_pdf_report(
            raw_input, 
            result, 
            explanation_list=shap_list, 
            shap_plot=shap_plot_path
        )
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Download Prediction Report (PDF)",
                data=f,
                file_name="fraud_prediction_report.pdf",
                mime="application/pdf"
            )

        # --- SHAP Waterfall Plot ---
        with st.expander("📊 Explain Prediction with SHAP"):
            st.info("This chart shows how each feature affected the model's decision.")

            # Get model and input
            model, shap_input = get_shap_input(raw_input)

            # SHAP explainer
            explainer = shap.Explainer(model)
            shap_values = explainer(shap_input)
            shap_val_row = shap_values[0].values
            expected_value = explainer.expected_value

            shap_df = pd.DataFrame({
                'Feature': shap_input.columns,
                'SHAP Value': shap_val_row
            }).sort_values(by='SHAP Value', key=abs, ascending=False).head(10)

            fig = px.bar(
                shap_df,
                x='SHAP Value',
                y='Feature',
                orientation='h',
                title='Top 10 Feature Contributions to Prediction',
                color='SHAP Value',
                color_continuous_scale='RdBu',
            )
            st.plotly_chart(fig, use_container_width=True)


        # --- SHAP Natural Language Explanation ---

        # Generate explanation
        explanations = explain_shap(raw_input.iloc[0], shap_val_row, shap_input.columns)

        st.markdown("### 🧠 What Likely Triggered This Prediction")
        for line in explanations:
            st.markdown(line)


        st.markdown("---")  
        st.markdown(
            "<p style='text-align: center; color: grey;'>This system uses a stacked ensemble of Random Forest, XGBoost, and LightGBM with a Logistic Regression meta-classifier for optimal fraud detection performance.</p>",
            unsafe_allow_html=True
        )

        # --- Save to transaction history ---
        
        history_file = "single_transaction_history.csv"

        # Add current date/time to the record
        result_df["Analyzed_At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Function to create a hash of the row (excluding timestamp)
        def hash_row(row, exclude_cols=["Analyzed_At"]):
            relevant_data = row.drop(labels=exclude_cols).astype(str).tolist()
            row_string = "|".join(relevant_data)
            return hashlib.md5(row_string.encode()).hexdigest()

        # Hash for the new transaction
        new_hash = hash_row(result_df.iloc[0])

        # If history file exists, check for duplicates
        if os.path.exists(history_file):
            history_df = pd.read_csv(history_file)

            # Generate hashes for existing rows
            existing_hashes = history_df.apply(lambda row: hash_row(row), axis=1).tolist()

            # Append only if not a duplicate
            if new_hash not in existing_hashes:
                result_df.to_csv(history_file, mode='a', index=False, header=False)
        else:
            result_df.to_csv(history_file, index=False)
            

# --- PAGE 2: BATCH ANALYSIS ---

elif selected_page == "Batch Analysis":
    # --- Header ---
    st.markdown("<h1 style='text-align: center;'>🛡️ Fraud Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: grey;'>Powered by Ensemble Machine Learning</h5>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.title("📁 Batch Fraud Detection")

    # --- Sample CSV Template ---
    st.markdown("### 🧾 Need help? Download the sample template")

    # Define sample template DataFrame
    sample_data = {
        "Transaction Amount": [100.0],
        "Quantity": [2],
        "Customer Age": [30],
        "Account Age Days": [365],
        "Transaction Hour": [14],
        "Transaction Date": ["2024-06-18"],    
        "Payment Method": ["credit card"],   
        "Product Category": ["electronics"],  
        "Device Used": ["mobile"],           
        "Customer Location": ["Malaysia"]           
    }
    sample_df = pd.DataFrame(sample_data)

    # Encode as CSV for download
    csv = sample_df.to_csv(index=False).encode('utf-8')

    # Download button
    st.download_button(
        label="⬇️ Download Sample CSV Template",
        data=csv,
        file_name='sample_transaction_template.csv',
        mime='text/csv',
        help="Click to download a sample template to ensure correct format."
    )

    # --- Batch Prediction via CSV Upload ---
    st.markdown("---")
    st.subheader("📁 Upload CSV for Batch Fraud Prediction")

    uploaded_file = st.file_uploader("Upload a CSV file with transactions", type=["csv"])

    if uploaded_file is not None:
        if "batch_df" not in st.session_state or st.session_state.get("last_uploaded_name") != uploaded_file.name:
            try:
                df = pd.read_csv(uploaded_file)

                st.session_state["last_uploaded_name"] = uploaded_file.name
                st.session_state["raw_batch_df"] = df.copy()  # keep a copy of original

                st.success(f"✅ {len(df)} transactions loaded successfully!")
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())

                with st.spinner("🔄 Processing and predicting all transactions..."):
                    predictions = []
                    probabilities = []
                    model_probs_list = []

                    for _, row in df.iterrows():
                        row_df = pd.DataFrame([row])
                        result = predict_fraud(row_df)
                        predictions.append("Fraudulent" if result["prediction"] == 1 else "Legitimate")
                        probabilities.append(round(result["probability"] * 100, 2))
                        model_probs_list.append(result["model_probs"])

                    df["Fraud_Prediction"] = predictions
                    df["Fraud_Probability (%)"] = [f"{p:.1f}%" for p in probabilities]

                    st.session_state["batch_df"] = df
                    st.session_state["fraud_count"] = df["Fraud_Prediction"].value_counts().get("Fraudulent", 0)
                    st.session_state["legit_count"] = df["Fraud_Prediction"].value_counts().get("Legitimate", 0)

                    st.success("✅ Predictions completed!")

                    # Create two columns (1/3 width for chart, 2/3 for summary)
                    col1, col2 = st.columns([1, 2])  # Ratio: 1 part chart, 2 parts text


                    #Generate Pie Chart
                    with col1:
                        fraud_count = st.session_state["fraud_count"]
                        legit_count = st.session_state["legit_count"]

                        labels = ["Fraudulent", "Legitimate"]
                        values = [fraud_count, legit_count]
                        colors = ["#ff4b4b", "#28a745"]

                        fig = go.Figure(data=[go.Pie(
                            labels=labels,
                            values=values,
                            marker=dict(colors=colors),
                            hole=0.5,
                            pull=[0.1, 0],  # Explode fraudulent slice
                            textinfo='label+percent',
                            insidetextfont=dict(color='white', size=14),
                            hoverinfo='label+percent+value'
                        )])

                        fig.update_layout(
                            showlegend=False,
                            margin=dict(t=0, b=0, l=0, r=0),
                            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                            plot_bgcolor='rgba(0,0,0,0)'
                        )

                        st.plotly_chart(fig, use_container_width=False)

                    # Generate Summary
                    with col2:
                        st.markdown("### 📊 Fraud Summary")
                        st.markdown(f"🛑 <b>Fraudulent:</b> <span style='color:red;'>{st.session_state["fraud_count"]}</span>", unsafe_allow_html=True)
                        st.markdown(f"✅ <b>Legitimate:</b> <span style='color:green;'>{st.session_state["legit_count"]}</span>", unsafe_allow_html=True)
                        st.markdown(f"🔢 <b>Total Transactions:</b> {st.session_state["fraud_count"] + st.session_state["legit_count"]}</span>", unsafe_allow_html=True)

                    # --- Custom Styling: Font color for 'Fraud_Prediction' column only ---
                    def style_prediction(val):
                        if val == "Fraudulent":
                            return "color: red; font-weight: bold;"
                        else:
                            return "color: green;"

                    styled_df = df.style.applymap(style_prediction, subset=["Fraud_Prediction"])
                    st.dataframe(styled_df, use_container_width=True)

                    # --- Download Button for CSV Results ---
                    batch_csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Batch Prediction Results (CSV)",
                        data=batch_csv,
                        file_name="batch_fraud_predictions.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"⚠️ Error reading or processing file: {e}")
                st.stop()

        else:
            df = st.session_state["batch_df"]
            st.success(f"✅ {len(df)} transactions loaded successfully!")
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            st.success("✅ Predictions completed!")
            st.write(f"🔢 Total Transactions: {len(df)}")
            st.write(f"🛑 Fraudulent: {st.session_state["fraud_count"]} | ✅ Legitimate: {st.session_state["legit_count"]}")

            # Create two columns (1/3 width for chart, 2/3 for summary)
            col1, col2 = st.columns([1, 2])  # Ratio: 1 part chart, 2 parts text

            with col1:
                # Generate Pie Chart
                fraud_count = st.session_state["fraud_count"]
                legit_count = st.session_state["legit_count"]

                labels = ["Fraudulent", "Legitimate"]
                values = [fraud_count, legit_count]
                colors = ["#ff4b4b", "#28a745"]

                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=colors),
                    hole=0.5,
                    pull=[0.1, 0],  # Explode fraudulent slice
                    textinfo='label+percent',
                    insidetextfont=dict(color='white', size=14),
                    hoverinfo='label+percent+value'
                )])

                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig, use_container_width=False)

            with col2:
                st.markdown("### 📊 Fraud Summary")
                st.markdown(f"🛑 <b>Fraudulent:</b> <span style='color:red;'>{st.session_state["fraud_count"]}</span>", unsafe_allow_html=True)
                st.markdown(f"✅ <b>Legitimate:</b> <span style='color:green;'>{st.session_state["legit_count"]}</span>", unsafe_allow_html=True)
                st.markdown(f"🔢 <b>Total Transactions:</b> {st.session_state["fraud_count"] + st.session_state["legit_count"]}</span>", unsafe_allow_html=True)

            # --- Custom Styling: Font color for 'Fraud_Prediction' column only ---
            def style_prediction(val):
                if val == "Fraudulent":
                    return "color: red; font-weight: bold;"
                else:
                    return "color: green;"

            styled_df = df.style.applymap(style_prediction, subset=["Fraud_Prediction"])
            st.dataframe(styled_df, use_container_width=True)

            # --- Download Button for CSV Results ---
            batch_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Batch Prediction Results (CSV)",
                data=batch_csv,
                file_name="batch_fraud_predictions.csv",
                mime="text/csv"
            )

# --- PAGE 3: View Transaction History ---

elif selected_page == "Transaction History":

    # --- Header ---
    st.markdown("<h1 style='text-align: center;'>🛡️ Fraud Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: grey;'>Powered by Ensemble Machine Learning</h5>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("## 📜 Transaction History")
    st.markdown("---")


    history_file = "single_transaction_history.csv"

    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)

        st.subheader("Full History Log")

        # Sort by most recent
        history_df = history_df.sort_values(by="Analyzed_At", ascending=False)

        # --- Format Fraud Probability as Percentage ---
        if "Fraud_Probability" in history_df.columns:
            if history_df["Fraud_Probability"].dtype != object:
                history_df["Fraud Probability"] = (history_df["Fraud_Probability"] * 100).round(2).astype(str) + "%"
            else:
                history_df["Fraud Probability"] = history_df["Fraud_Probability"]

        # Rename the prediction column
        if "Fraud_Prediction" in history_df.columns:
            history_df.rename(columns={"Fraud_Prediction": "Fraud Prediction"}, inplace=True)

        # Drop the original probability column (raw float)
        history_df.drop(columns=["Fraud_Probability"], inplace=True, errors="ignore")

        # Reorder columns to match your specified order
        desired_order = [
            "Transaction Amount", "Quantity", "Customer Age", "Account Age Days", "Transaction Hour",
            "Payment Method", "Product Category", "Device Used", "Customer Location",
            "Transaction Year", "Transaction Month", "Transaction Day",
            "Fraud Prediction", "Fraud Probability", "Analyzed_At"
        ]
        # Only keep columns that exist to avoid KeyError
        ordered_columns = [col for col in desired_order if col in history_df.columns]
        history_df = history_df[ordered_columns]

        # --- Style function for Fraud Prediction ---
        def color_prediction(val):
            if val == "Fraudulent":
                return "color: red; font-weight: bold"
            elif val == "Legitimate":
                return "color: green; font-weight: bold"
            return ""

        # Apply style
        styled_df = history_df.style.applymap(color_prediction, subset=["Fraud Prediction"])

        # Show styled DataFrame
        st.dataframe(styled_df, use_container_width=True)

        # --- Download button ---
        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name="single_transaction_history.csv",
            mime="text/csv"
        )

        # --- Clear History Button ---
        if st.button("🗑️ Clear History", help="Delete all stored transaction history. This action cannot be undone."):
            if os.path.exists(history_file):
                os.remove(history_file)
            st.session_state.pop("raw_input", None)
            st.session_state.pop("result", None)
            st.session_state.pop("fraud_prob", None)
            st.session_state.pop("predict_triggered", None)
            st.success("History cleared successfully.")
            st.rerun()

        st.markdown("---")
        
        if "Fraud Prediction" in history_df.columns:
            total_txns = len(history_df)
            fraud_count = history_df["Fraud Prediction"].value_counts().get("Fraudulent", 0)
            legit_count = history_df["Fraud Prediction"].value_counts().get("Legitimate", 0)
            fraud_rate = (fraud_count / total_txns) * 100 if total_txns > 0 else 0
            legit_rate = (legit_count / total_txns) * 100 if total_txns > 0 else 0

            # Donut Chart
            fig = go.Figure(data=[go.Pie(
                labels=["Fraudulent", "Legitimate"],
                values=[fraud_count, legit_count],
                hole=0.5,
                pull=[0.1, 0],
                marker=dict(colors=["#ff4b4b", "#28a745"]),
                textinfo="label+percent",
                insidetextfont=dict(color='white', size=14),
                hoverinfo='label+percent+value'
            )])

            fig.update_layout(
                width=600,
                height=400,
                showlegend=False,
                legend=dict(
                    font=dict(size=14),
                    orientation="v",
                    yanchor="middle",
                    xanchor="left",
                    x=1.05,
                    y=0.5
                ),
                margin=dict(t=30, b=0, l=0, r=150),
                paper_bgcolor="rgba(0,0,0,0)",  # transparent bg for dark mode
                plot_bgcolor="rgba(0,0,0,0)"
            )

            # Render chart in Streamlit
            st.subheader("📊 Transaction History Summary")

            col1, col2 = st.columns([1, 3])
            with col1:
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                # Displaying stats alongside
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown(f"🛑 <b>Fraudulent:</b> <span style='color:red;'>{fraud_count}</span>", unsafe_allow_html=True)
                st.markdown(f"✅ <b>Legitimate:</b> <span style='color:green;'>{legit_count}</span>", unsafe_allow_html=True)
                st.markdown(f"🔢 <b>Total Transactions:</b> {total_txns}</span>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("📈 Daily Transaction Trend Over Time")

            # Ensure Analyzed_At is datetime
            history_df["Analyzed_At"] = pd.to_datetime(history_df["Analyzed_At"], errors="coerce")

            # Drop rows with invalid timestamps
            history_df = history_df.dropna(subset=["Analyzed_At"])

            # Group by date and prediction type
            history_df["Date"] = history_df["Analyzed_At"].dt.date
            daily_summary = history_df.groupby(["Date", "Fraud Prediction"]).size().unstack(fill_value=0)

            # Ensure consistent ordering
            daily_summary = daily_summary.sort_index()

            fig = px.line(
                daily_summary,
                x=daily_summary.index,
                y=daily_summary.columns,
                color_discrete_map={
                    "Fraudulent": "#ff4b4b",
                    "Legitimate": "#28a745"
                },
                markers=True,
                labels={"value": "Transactions", "variable": "Type", "x": "Date"}
            )
            fig.update_layout(
                font=dict(color="white"),
            )
            
            # Plot with Streamlit line chart
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No history found yet. Analyze at least one transaction to see results here.")     
    

# --- PAGE 4: ABOUT ---
elif selected_page == "About":
    st.title("🛡️ Multi-Layered Fraud Detection System")
    st.markdown("---")

    # Welcome Message

    st.subheader("👋 Welcome to the Fraud Detection System")
    st.markdown("""
    This intelligent application is built to detect **potentially fraudulent e-commerce transactions** using a **multi-layered ensemble machine learning framework**.

    🧠 It integrates several powerful models to provide **reliable predictions**, backed by **explainable AI** with SHAP visualizations.

    🔍 Whether you're analyzing a **single transaction** or processing a **batch of thousands**, this tool empowers you to:
    - Evaluate risk probabilities
    - Understand model decision rationale
    - Export reports in CSV & PDF formats
    - Apply fraud insights in real time

    📌 **Use Case Example**  
    A transaction at **3:00 AM**, involving **high-value electronics**, from a **new account**—even with a familiar payment method—might raise red flags. This app helps detect such anomalies **before** losses occur.

    ---
    """)

    st.success("✨ Did you know? Ensemble learning can improve fraud detection accuracy by over 15% compared to single models.")

    with st.expander("💡 Why This Matters"):
        st.markdown("""
        E-commerce fraud costs companies **billions annually**.  
        By combining machine learning with explainability (SHAP), this app helps:
        - **Prevent financial loss**
        - **Support compliance audits**
        - **Improve customer trust**
        """)
        


    st.markdown("---")

    # Ensemble Learning Overview Section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📈 What is Ensemble Learning?")
        st.markdown("""
        **Ensemble learning** is a machine learning method that combines the predictions of multiple models to improve overall performance.  
        Rather than relying on a single algorithm, ensemble techniques reduce bias, variance, and overfitting.

        ### 🧱 Stacking Ensemble (Used in This App)
        This app uses a technique called **stacking**, where:
        - **Base models** (Random Forest, XGBoost, LightGBM) are trained separately.
        - Their output probabilities are **combined using a meta-model** (Logistic Regression).
        - This **meta-model** learns how to best weight and combine those predictions for a stronger final decision.

        > ✅ **Why stacking?** It harnesses the unique strengths of each model to deliver more robust and accurate fraud detection.
        """)
        
    # with col2:
    #     st.image("assets/meta_model.png", caption="Ensemble Stacking Flowchart", use_container_width=False)


    st.markdown("---")

    # ML Models Section with Expanders
    st.subheader("🧠 Machine Learning Models Used")

    with st.expander("🌳 Random Forest"):

        tab1, tab2, tab3, tab4 = st.tabs(["About", "Accuracy", "Feature Importance", "Confusion Matrix"])

        with tab1:
            st.markdown("""
            Random Forest is an ensemble method that builds multiple decision trees and merges their results to improve accuracy and control overfitting.  
            - **Strengths**: Robust to noise, handles missing values, works well with mixed feature types.  
            - **Why used**: Great baseline model and helps the ensemble capture nonlinear interactions.
            """)
        with tab2:
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Accuracy", "95%")
            r2.metric("Precision", "98%")
            r3.metric("Recall", "92%")
            r4.metric("F1 Score", "95%")
            r5.metric("AUC-ROC", "0.979")
            style_metric_cards(background_color="#262730", border_left_color="#1f77b4", border_size_px=1)
        with tab3:
            st.plotly_chart(plot_feature_importance(feature_importance_rf, "Random Forest Feature Importance"))
        with tab4:
            # Define the specific data for this chart
            rf_cm_data = [
                [98041, 1959], 
                [8448, 91552]   
            ]
            class_labels = ['Legitimate', 'Fraudulent']

            # Call the function to create the figure
            fig = create_interactive_confusion_matrix(
                cm_data=rf_cm_data,
                labels=class_labels,
                title="Random Forest Confusion Matrix"
            )

            # Display the figure in Streamlit
            st.plotly_chart(fig, use_container_width=False)

    with st.expander("⚡ XGBoost"):

        tab1, tab2, tab3, tab4 = st.tabs(["About", "Accuracy", "Feature Importance", "Confusion Matrix"])
        
        with tab1:
            st.markdown("""
            XGBoost (Extreme Gradient Boosting) is a highly efficient gradient boosting algorithm.  
            - **Strengths**: Regularization to prevent overfitting, handles sparse data well.  
            - **Why used**: Often provides state-of-the-art performance on tabular datasets.
            """)
        with tab2:
            x1, x2, x3, x4, x5 = st.columns(5)
            x1.metric("Accuracy", "95%")
            x2.metric("Precision", "99%")
            x3.metric("Recall", "91%")
            x4.metric("F1 Score", "95%")
            x5.metric("AUC-ROC", "0.979")
            style_metric_cards(background_color="#262730", border_left_color="#1f77b4", border_size_px=1)
        with tab3:
            st.plotly_chart(plot_feature_importance(feature_importance_xgb, "XGBoost Feature Importance"))
        with tab4:
            # Define the specific data for this chart
            xgb_cm_data = [
                [98680, 1320],  
                [8698, 91302]  
            ]
            class_labels = ['Legitimate', 'Fraudulent']

            # Call the function to create the figure
            fig = create_interactive_confusion_matrix(
                cm_data=xgb_cm_data,
                labels=class_labels,
                title="XGBoost Confusion Matrix"
            )

            # Display the figure in Streamlit
            st.plotly_chart(fig, use_container_width=False)

    with st.expander("🔦 LightGBM"):

        tab1, tab2, tab3, tab4 = st.tabs(["About", "Accuracy", "Feature Importance", "Confusion Matrix"])

        with tab1:
            st.markdown("""
            LightGBM (Light Gradient Boosting Machine) is optimized for speed and performance.  
            - **Strengths**: Faster training, supports large datasets, works well with categorical data.  
            - **Why used**: Complements the ensemble with high-speed, high-accuracy predictions.
            """)
        with tab2:
            l1, l2, l3, l4, l5 = st.columns(5)
            l1.metric("Accuracy", "95%")
            l2.metric("Precision", "99%")
            l3.metric("Recall", "91%")
            l4.metric("F1 Score", "95%")
            l5.metric("AUC-ROC", "0.980")
            style_metric_cards(background_color="#262730", border_left_color="#1f77b4", border_size_px=1)
        with tab3:
            st.plotly_chart(plot_feature_importance(feature_importance_lgbm, "LightGBM Feature Importance"))
        with tab4:
            # Define the specific data for this chart
            lgbm_cm_data = [
                [98731, 1269],  
                [8715, 91285]  
            ]
            class_labels = ['Legitimate', 'Fraudulent']

            # Call the function to create the figure
            fig = create_interactive_confusion_matrix(
                cm_data=lgbm_cm_data,
                labels=class_labels,
                title="LightGBM Confusion Matrix"
            )

            # Display the figure in Streamlit
            st.plotly_chart(fig, use_container_width=False)


    with st.expander("➕ Logistic Regression (Meta-Model)"):

        tab1, tab2, tab3, tab4 = st.tabs(["About", "Accuracy", "SHAP Impact", "Confusion Matrix"])

        with tab1:
            st.markdown("""
            Logistic Regression is used as the **meta-model** in this stacked ensemble.  
            - **Role**: Combines predictions from base models and learns optimal weights for final decision-making.  
            - **Why used**: Simple yet effective, avoids overfitting and interpretable.
            """)
        with tab2:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", "95%")
            m2.metric("Precision", "97%")
            m3.metric("Recall", "93%")
            m4.metric("F1 Score", "95%")
            m5.metric("AUC-ROC", "0.980")

            style_metric_cards(background_color="#262730", border_left_color="#1f77b4", border_size_px=1)
        with tab3:
            st.plotly_chart(plot_shap_importance(shap_importance_logreg, "Meta-Model SHAP Summary"))
        with tab4:
            # Define the specific data for this chart
            logreg_cm_data = [
                [97133, 2867],  # True: Legitimate
                [7236, 92764]   # True: Fraudulent
            ]
            class_labels = ['Legitimate', 'Fraudulent']

            # Call the function to create the figure
            fig = create_interactive_confusion_matrix(
                cm_data=logreg_cm_data,
                labels=class_labels,
                title="Meta-Model Confusion Matrix"
            )

            # Display the figure in Streamlit
            st.plotly_chart(fig, use_container_width=False)

    st.markdown("---")

    # Key Features Section
    st.subheader("🔍 App Capabilities")
    st.markdown("""
    - ✅ Real-time fraud prediction (form-based)
    - 📁 Batch fraud prediction via CSV
    - 📊 SHAP explainability plots
    - 📄 PDF and CSV report generation
    - 📜 Transaction History    
    - 📈 Interactive dashboard summaries

    All features are designed to mimic real-world fraud detection systems for demonstration and learning.
    """)

    st.markdown("---")

    # Disclaimer
    st.warning("⚠️ Disclaimer: This tool is for educational and research use only. It is **not** intended for use in production financial systems.")