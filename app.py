import streamlit as st
import joblib


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="AI Spam Message Classifier",
    page_icon="🤖",
    layout="centered"
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model_path = "model/spam_classifier.pkl"

model = joblib.load(model_path)


# ==========================================
# TITLE
# ==========================================

st.title("🤖 AI Spam Message Classifier")

st.write(
    "Enter a message below and the AI model will "
    "predict whether it is Spam or Not Spam."
)


# ==========================================
# MESSAGE INPUT
# ==========================================

message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You won a free prize..."
)


# ==========================================
# CHECK BUTTON
# ==========================================

if st.button("🔍 Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message first.")

    else:

        prediction = model.predict([message])[0]

        st.subheader("Prediction")

        if prediction == "spam":

            st.error("🚨 SPAM")

            st.write(
                "This message may be unwanted or suspicious."
            )

        else:

            st.success("✅ NOT SPAM")

            st.write(
                "This message appears to be normal."
            )


# ==========================================
# PROJECT INFORMATION
# ==========================================

st.divider()

st.caption(
    "SWYNEX AI Problem Design | "
    "Spam Message Classification"
)