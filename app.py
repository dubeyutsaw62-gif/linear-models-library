import streamlit as st
from transformers import pipeline

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="Zero-Shot Spam Guard", page_icon="🧠")

# 1. Load the Zero-Shot Reasoning Model
@st.cache_resource
def load_zero_shot():
    # BART-large is a reasoning engine. 
    # Note: It is a large model (~1.6GB), so the first run will take a minute to download!
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_zero_shot()

# --- USER INTERFACE ---
st.title("🧠 Zero-Shot AI Spam Guard")
st.markdown("""
This prototype doesn't rely on narrow spam datasets. It uses **Zero-Shot Learning** to logically deduce if a message matches the definition of a scam or a normal conversation.
""")

user_input = st.text_area("Enter the message you want to check:", placeholder="Type your message here...", height=150)

if st.button("Analyze with AI"):
    if user_input.strip():
        with st.spinner("Reasoning through the context..."):
            
            # 2. Define the exact categories we want the AI to choose between
            categories = [
                "a manipulative marketing scam or malicious spam", 
                "a normal, casual, safe text message between people"
            ]
            
            # 3. The AI reads the text and assigns a probability to each category
            result = classifier(user_input, candidate_labels=categories)
            
            # The results are sorted by highest confidence automatically
            winning_label = result['labels'][0]
            winning_score = result['scores'][0]
            
            # --- DISPLAY RESULTS ---
            st.write("---")
            if "scam" in winning_label:
                st.error(f"🚨 **RESULT: SPAM** (Confidence: {winning_score:.2%})")
                st.info("The AI logically matched this to manipulative or promotional behavior.")
            else:
                st.success(f"✅ **RESULT: HAM (Safe)** (Confidence: {winning_score:.2%})")
                st.info("The AI recognizes this as standard, safe human conversation.")
                
    else:
        st.warning("Please enter some text first!")