import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="IMDb Movie Gross Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Dark CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    label, p, span, h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }

    div[data-testid="stColumn"], div[data-testid="stExpander"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 16px;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #E50914 0%, #B81D24 100%);
        color: #FFFFFF !important;
        font-weight: bold !important;
        font-size: 18px !important;
        border-radius: 8px !important;
        padding: 0.6rem !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.4);
    }

    .result-card {
        background: #1F242D;
        border: 2px solid #FFD700;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }

    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        border-top: 1px solid #30363D;
        color: #8B949E !important;
        font-size: 14px;
    }

    .footer a {
        color: #FFD700 !important;
        text-decoration: none;
        font-weight: bold;
    }

    .footer a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Load Model and Scaler
@st.cache_resource
def load_artifacts():
    with open("imdb_ann_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception:
    st.error("Model files failed to load. Please check file paths.")

def format_inr(number):
    if number >= 10000007:
        return f"₹{number / 10000000:,.2f} Cr"
    elif number >= 100000:
        return f"₹{number / 100000:,.2f} Lakh"
    return f"₹{number:,.2f}"

# 3. Sidebar Section
with st.sidebar:
    st.title("🎬 CineMetrics AI")
    st.caption("Advanced Box Office Intelligence Platform")
  
    
    st.subheader("💱 Currency Settings")
    currency = st.radio("Display Currency", ["USD ($)", "INR (₹)"])
    usd_to_inr = 83.0  
    
    
    
    # English About App Section
    st.subheader("ℹ️ About CineMetrics AI")
    st.write(
        "An Machine Learning-powered analytics tool that predicts worldwide "
        "box office earnings of movies based on historical IMDb dataset."
    )
    
    st.markdown("**🧠 How it Works:**")
    st.markdown("""
        "• **Input Normalization:** Scaled using `StandardScaler`."
        "• **ANN Evaluation:** Multi-feature neural weights calculation."
        "• **Real-time Formatting:** Results are formatted instantly and displayed with dynamic currency conversion."
    """)
    
    st.markdown("**📌 Key Predictors:**")
    st.markdown("""
        "• **IMDb Score:** Public perception & audience rating."
        "• **Metascore:** Verified critic review score."
        "• **Runtime:** Total film duration."
        "• **Vote Volume:** Global user engagement & popularity."
    """)

# 4. Main Page Header
st.title("🎬 IMDb Movie Gross Predictor")
st.caption("AI-powered Box Office Revenue Forecasting")
st.markdown("---")

# Presets Section
st.subheader("⚡ Quick Load Presets")
preset = st.selectbox(
    "Select a pre-filled movie scenario:",
    ["Custom Input", "Blockbuster Action (e.g., Inception)", "Critically Acclaimed Drama", "Indie Low-Budget"]
)

def_rating, def_meta, def_dur, def_votes = 8.0, 75, 120, 100000

if preset == "Blockbuster Action (e.g., Inception)":
    def_rating, def_meta, def_dur, def_votes = 8.8, 74, 148, 2400000
elif preset == "Critically Acclaimed Drama":
    def_rating, def_meta, def_dur, def_votes = 8.5, 92, 130, 500000
elif preset == "Indie Low-Budget":
    def_rating, def_meta, def_dur, def_votes = 6.8, 60, 95, 25000

# Inputs Section
st.subheader("⚙️ Movie Features")
col1, col2 = st.columns(2, gap="medium")

with col1:
    imdb_rating = st.slider("⭐ IMDb Rating", 1.0, 10.0, float(def_rating), 0.1)
    metascore = st.slider("🎯 Metascore", 0, 100, int(def_meta), 1)

with col2:
    duration = st.number_input("⏱️ Duration (Minutes)", 30, 300, int(def_dur), 5)
    votes = st.number_input("🗳️ Votes Count", 1000, 3000000, int(def_votes), 5000)

st.markdown("<br>", unsafe_allow_html=True)

# 5. Prediction Logic
if st.button("🚀 PREDICT BOX OFFICE GROSS"):
    raw_input_data = np.array([[imdb_rating, metascore, duration, votes]])
    scaled_data = scaler.transform(raw_input_data)
    
    prediction = model.predict(scaled_data)[0]
    final_gross = max(0, prediction)

    if currency == "INR (₹)":
        display_val = format_inr(final_gross * usd_to_inr)
    else:
        display_val = f"${final_gross:,.2f}"

    st.markdown(f"""
        <div class="result-card">
            <p style="color: #A0A0A0; font-size: 16px; margin: 0;">ESTIMATED GROSS COLLECTION</p>
            <h1 style="color: #00FF66; font-size: 40px; margin: 5px 0;">{display_val}</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Normalized Input Scale")
    
    categories = ['IMDb Rating', 'Metascore', 'Duration (Mins)', 'Votes (in 10k)']
    norm_values = [
        imdb_rating * 10,
        metascore,
        (duration / 300) * 100,
        min((votes / 1000000) * 100, 100)
    ]

    fig = go.Figure(data=[
        go.Bar(
            x=categories, 
            y=norm_values, 
            marker_color=['#FFD700', '#FF4B4B', '#00D2FF', '#00FF66']
        )
    ])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        yaxis=dict(title="Normalized Scale (0-100)", gridcolor='#30363D'),
        xaxis=dict(gridcolor='#30363D'),
        height=320,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    if final_gross > 100_000_000:
        st.balloons()
        st.success("🎉 **Blockbuster Alert!** Movie predicted to cross major revenue records.")

# 6. Footer Section
st.markdown("""
    <div class="footer">
        Developed with ❤️ using Streamlit & Machine Learning | 
        <a href="https://github.com/AishAftab098" target="_blank">GitHub Repository</a> • 
        <a href="https://www.linkedin.com/in/aish-aftab" target="_blank">LinkedIn Profile</a>
    </div>
""", unsafe_allow_html=True)