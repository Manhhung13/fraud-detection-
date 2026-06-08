from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# app/streamlit_app.py -> parents[1] = fraud-detection-project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.inference.predictor import load_final_predictor


st.set_page_config(
    page_title="Fraud Detection Demo",
    layout="wide",
)

st.title("Fraud Detection Demo")
st.caption("Demo dùng final CatBoost model. Form hiện tại nhập feature-ready values.")

predictor = load_final_predictor(PROJECT_ROOT)

with st.sidebar:
    st.header("Model info")
    st.write("Experiment:", predictor.metadata.get("final_experiment_name"))
    st.write("Threshold:", predictor.threshold)
    st.write("Model:", predictor.metadata.get("model_name"))


st.subheader("1. Nhập feature giao dịch")

with st.form("fraud_prediction_form"):
    st.markdown("### Behavioral numeric features")

    col1, col2, col3 = st.columns(3)

    with col1:
        mcc_entropy_30d = st.number_input("mcc_entropy_30d", value=0.40, step=0.01)
        night_ratio_30d = st.number_input("night_ratio_30d", value=0.10, step=0.01)
        log_amount_to_max_30d = st.number_input("log_amount_to_max_30d", value=0.80, step=0.01)
        spending_trend = st.number_input("spending_trend", value=0.60, step=0.01)

    with col2:
        distinct_countries_30d = st.number_input("distinct_countries_30d", value=3.0, step=1.0)
        amount_z_30d = st.number_input("amount_z_30d", value=1.50, step=0.10)
        decline_rate_30d = st.number_input("decline_rate_30d", value=0.20, step=0.01)
        device_diversity_30d = st.number_input("device_diversity_30d", value=10.0, step=1.0)

    with col3:
        credit_util_today = st.number_input("credit_util_today", value=0.50, step=0.01)
        log_mean_amount_30d = st.number_input("log_mean_amount_30d", value=8.0, step=0.10)
        distinct_merchants_7d = st.number_input("distinct_merchants_7d", value=4.0, step=1.0)
        txn_count_ratio_7d_30d = st.number_input("txn_count_ratio_7d_30d", value=0.50, step=0.01)

    st.markdown("### Additional numeric features")

    col4, col5, col6 = st.columns(3)

    with col4:
        txn_count_7d = st.number_input("txn_count_7d", value=5.0, step=1.0)
        txn_count_30d = st.number_input("txn_count_30d", value=20.0, step=1.0)
        amount_to_mean_30d = st.number_input("amount_to_mean_30d", value=1.20, step=0.10)

    with col5:
        hour = st.number_input("hour", value=14.0, min_value=0.0, max_value=23.0, step=1.0)
        ip_score = st.number_input("ip_score", value=0.30, step=0.01)
        chargebacks_365d = st.number_input("chargebacks_365d", value=0.0, step=1.0)

    with col6:
        night_unusual_score = st.number_input("night_unusual_score", value=0.0, step=0.01)

    st.markdown("### Binary flags")

    col7, col8, col9 = st.columns(3)

    with col7:
        cross_border_flag = st.selectbox("cross_border_flag", [0, 1], index=0)
        high_max_amount_30d_flag = st.selectbox("high_max_amount_30d_flag", [0, 1], index=0)
        low_country_diversity_flag = st.selectbox("low_country_diversity_flag", [0, 1], index=0)

    with col8:
        card_not_present_flag = st.selectbox("card_not_present_flag", [0, 1], index=0)
        very_low_mcc_entropy_flag = st.selectbox("very_low_mcc_entropy_flag", [0, 1], index=0)
        low_spending_trend_flag = st.selectbox("low_spending_trend_flag", [0, 1], index=0)

    with col9:
        low_mcc_entropy_flag = st.selectbox("low_mcc_entropy_flag", [0, 1], index=0)
        night_transaction_flag = st.selectbox("night_transaction_flag", [0, 1], index=0)
        tokenised_flag = st.selectbox("tokenised_flag", [0, 1], index=0)
        low_night_ratio_flag = st.selectbox("low_night_ratio_flag", [0, 1], index=0)

    st.markdown("### Categorical context")

    col10, col11, col12 = st.columns(3)

    with col10:
        mcc = st.text_input("mcc", value="infrequent_sklearn")
        currency = st.selectbox("currency", ["VND", "USD", "EUR", "JPY", "SGD", "GBP", "AUD", "CAD"])
        payment_channel = st.selectbox(
            "payment_channel",
            ["MOBILE_APP", "WEB_BROWSER", "POS_TERMINAL", "ATM", "SMART_WATCH", "API_SERVER", "IVR_PHONE"],
        )
        merchant_country = st.selectbox(
            "merchant_country",
            ["VN", "US", "SG", "TH", "IN", "JP", "GB", "FR", "DE", "CN", "HK", "AU", "CA"],
        )

    with col11:
        card_present = st.selectbox("card_present", ["true", "false"])
        card_entry_mode = st.selectbox(
            "card_entry_mode",
            ["TOKEN_IN_APP", "QR_CODE", "CHIP", "CONTACTLESS", "MANUAL_KEYED", "MAGSTRIPE"],
        )
        auth_result = st.selectbox(
            "auth_result",
            ["3DS_PASS", "CVV_PASS", "AVS_PASS", "BIOMETRIC_PASS", "3DS_FAIL", "CVV_FAIL", "BIOMETRIC_FAIL"],
        )
        pin_verif_method = st.selectbox(
            "pin_verif_method",
            ["NONE", "ONLINE_PIN", "OFFLINE_PIN", "SIGNATURE"],
        )

    with col12:
        tokenised = st.selectbox("tokenised", ["true", "false"])
        recurring_flag = st.selectbox(
            "recurring_flag",
            ["SINGLE_PURCHASE", "RECURRING_SUBSCRIPTION", "INSTALLMENT_PLAN", "MOTO_PHONE", "MOTO_MAIL"],
        )
        cross_border = st.selectbox("cross_border", ["true", "false"])
        auth_characteristics = st.selectbox(
            "auth_characteristics",
            ["E_COMMERCE", "CARD_PRESENT", "CRYPTOGRAM_PRESENT", "MOBILE_TOKEN", "FALLBACK_MAG", "FORCE_POST"],
        )

    col13, col14, col15 = st.columns(3)

    with col13:
        message_type = st.selectbox(
            "message_type",
            ["0100-AuthReq", "0110-AuthRsp", "0200-FinReq", "0210-FinRsp", "0400-Reversal", "0420-RepeatRev", "0500-Advice", "0510-AdviceRsp"],
        )
        term_location = st.selectbox(
            "term_location",
            ["MOBILE_APP", "ONLINE_PORTAL", "POS_TERMINAL", "WEARABLE_PAY", "CALL_CENTER", "BRANCH_COUNTER", "KIOSK"],
        )

    with col14:
        day_name = st.selectbox(
            "day_name",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        )
        time_period = st.selectbox(
            "time_period",
            ["morning", "afternoon", "evening", "night", "late_evening"],
        )

    with col15:
        amount_to_max_group = st.selectbox(
            "amount_to_max_group",
            ["below_half_max", "near_max", "above_max", "much_above_max", "extreme"],
        )
        night_ratio_group = st.selectbox(
            "night_ratio_group",
            ["very_low", "low", "medium", "high"],
        )
        credit_util_risk_group = st.selectbox(
            "credit_util_risk_group",
            ["low", "medium", "high", "very_high"],
        )

    col16, col17, col18 = st.columns(3)

    with col16:
        decline_rate_risk_group = st.selectbox(
            "decline_rate_risk_group",
            ["low", "medium", "high"],
        )

    with col17:
        ip_score_risk_group = st.selectbox(
            "ip_score_risk_group",
            ["low", "medium", "high", "very_high"],
        )

    with col18:
        txn_count_7d_group = st.selectbox(
            "txn_count_7d_group",
            ["0_1", "2_5", "6_10", "11_20", "gt_20"],
        )
        txn_velocity_group = st.selectbox(
            "txn_velocity_group",
            ["very_low", "low", "medium", "high", "very_high"],
        )
        night_unusual_group = st.selectbox(
            "night_unusual_group",
            ["not_night", "low", "medium", "high", "very_high"],
        )

    submitted = st.form_submit_button("Predict fraud risk")


if submitted:
    features = {
        # numeric
        "mcc_entropy_30d": mcc_entropy_30d,
        "night_ratio_30d": night_ratio_30d,
        "log_amount_to_max_30d": log_amount_to_max_30d,
        "spending_trend": spending_trend,
        "distinct_countries_30d": distinct_countries_30d,
        "amount_z_30d": amount_z_30d,
        "decline_rate_30d": decline_rate_30d,
        "device_diversity_30d": device_diversity_30d,
        "credit_util_today": credit_util_today,
        "log_mean_amount_30d": log_mean_amount_30d,
        "distinct_merchants_7d": distinct_merchants_7d,
        "txn_count_ratio_7d_30d": txn_count_ratio_7d_30d,
        "txn_count_7d": txn_count_7d,
        "txn_count_30d": txn_count_30d,
        "amount_to_mean_30d": amount_to_mean_30d,
        "hour": hour,
        "ip_score": ip_score,
        "chargebacks_365d": chargebacks_365d,
        "night_unusual_score": night_unusual_score,

        # binary
        "cross_border_flag": cross_border_flag,
        "high_max_amount_30d_flag": high_max_amount_30d_flag,
        "low_country_diversity_flag": low_country_diversity_flag,
        "card_not_present_flag": card_not_present_flag,
        "very_low_mcc_entropy_flag": very_low_mcc_entropy_flag,
        "low_spending_trend_flag": low_spending_trend_flag,
        "low_mcc_entropy_flag": low_mcc_entropy_flag,
        "night_transaction_flag": night_transaction_flag,
        "tokenised_flag": tokenised_flag,
        "low_night_ratio_flag": low_night_ratio_flag,

        # categorical
        "mcc": mcc,
        "currency": currency,
        "payment_channel": payment_channel,
        "merchant_country": merchant_country,
        "card_present": card_present,
        "card_entry_mode": card_entry_mode,
        "auth_result": auth_result,
        "pin_verif_method": pin_verif_method,
        "tokenised": tokenised,
        "recurring_flag": recurring_flag,
        "cross_border": cross_border,
        "auth_characteristics": auth_characteristics,
        "message_type": message_type,
        "term_location": term_location,
        "day_name": day_name,
        "time_period": time_period,
        "amount_to_max_group": amount_to_max_group,
        "night_ratio_group": night_ratio_group,
        "credit_util_risk_group": credit_util_risk_group,
        "decline_rate_risk_group": decline_rate_risk_group,
        "ip_score_risk_group": ip_score_risk_group,
        "txn_count_7d_group": txn_count_7d_group,
        "txn_velocity_group": txn_velocity_group,
        "night_unusual_group": night_unusual_group,
    }

    input_df = pd.DataFrame([features])
    result = predictor.predict(input_df).iloc[0].to_dict()

    st.subheader("2. Prediction result")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Fraud probability", f"{result['fraud_probability']:.4f}")

    with metric_col2:
        st.metric("Prediction", result["prediction"])

    with metric_col3:
        st.metric("Risk level", result["risk_level"])

    with metric_col4:
        st.metric("Decision", result["decision"])

    st.write("Threshold:", result["threshold"])
    st.info(result["reason"])

    st.markdown("### Raw output")
    st.json(result)

    st.markdown("### Input features")
    st.dataframe(input_df)