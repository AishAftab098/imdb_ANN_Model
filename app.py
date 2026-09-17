import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# -------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="IMDb Movie Gross Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------------------------------
# 2. CUSTOM PROFESSIONAL LIGHT THEME
# -------------------------------------------------
st.markdown("""
<style>

    /* Main App */
    .stApp {
        background-color: #f6f8fc;
        color: #172033;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #eef3fa;
        border-right: 1px solid #d9e2ef;
    }

    section[data-testid="stSidebar"] * {
        color: #1d2b44 !important;
    }

    /* General Text */
    h1, h2, h3, h4, h5, h6 {
        color: #172033 !important;
    }

    p, label, span {
        color: #34435b;
    }

    /* Hero Section */
    .hero-card {
        background: linear-gradient(
            110deg,
            #10284b 0%,
            #1d4778 55%,
            #376fa6 100%
        );
        padding: 32px;
        border-radius: 18px;
        margin-bottom: 25px;
        color: white !important;
        box-shadow: 0 8px 25px rgba(36, 70, 110, 0.15);
    }

    .hero-card h1 {
        color: white !important;
        font-size: 42px;
        margin-bottom: 8px;
        font-weight: 750;
    }

    .hero-card p {
        color: #e6efff !important;
        font-size: 17px;
        margin-bottom: 0;
    }

    .hero-badge {
        display: inline-block;
        background-color: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 6px 13px;
        border-radius: 20px;
        font-size: 13px;
        color: white !important;
        margin-bottom: 15px;
    }

    /* Section Cards */
    .section-card {
        background-color: white;
        border: 1px solid #dce5f0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 5px 18px rgba(38, 64, 98, 0.06);
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #1d3557 !important;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #718096 !important;
        font-size: 14px;
        margin-bottom: 18px;
    }

    /* Inputs */
    div[data-baseweb="input"] {
        border-radius: 9px;
    }

    div[data-baseweb="select"] > div {
        border-radius: 9px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background: #2f65a7;
        color: white !important;
        border: none;
        border-radius: 9px;
        padding: 12px 20px;
        font-size: 17px;
        font-weight: 650;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: #234f88;
        border: none;
        transform: translateY(-1px);
    }

    /* Prediction Result */
    .result-card {
        background: linear-gradient(135deg, #effaf5, #f8fffb);
        border: 1px solid #b9e6cf;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .result-label {
        color: #438064 !important;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .result-value {
        color: #168653 !important;
        font-size: 38px;
        font-weight: 800;
        margin: 5px 0;
    }

    .result-note {
        color: #5f806f !important;
        font-size: 13px;
    }

    /* Info Box */
    .info-box {
        background-color: #f0f6ff;
        border: 1px solid #c9def8;
        border-radius: 10px;
        padding: 14px;
        color: #315b87 !important;
        font-size: 13px;
        margin-top: 15px;
    }

    /* Sidebar Card */
    .sidebar-card {
        background-color: white;
        border: 1px solid #d8e3f0;
        border-radius: 12px;
        padding: 15px;
        margin-top: 18px;
    }

    .sidebar-card h4 {
        color: #1d3557 !important;
        font-size: 16px;
        margin-bottom: 8px;
    }

    .sidebar-card p {
        color: #5c6b80 !important;
        font-size: 13px;
        line-height: 1.6;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 25px 10px 10px 10px;
        margin-top: 35px;
        border-top: 1px solid #dce5f0;
        color: #78869a !important;
        font-size: 13px;
    }

    .footer a {
        color: #2f65a7 !important;
        text-decoration: none;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# 3. LOAD MODEL AND SCALER
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("imdb_ann_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    return model, scaler


try:
    model, scaler = load_artifacts()
except Exception as error:
    st.error("Model files could not be loaded. Please check the model file names.")
    st.stop()


# -------------------------------------------------
# 4. HELPER FUNCTIONS
# -------------------------------------------------
def format_inr(number):
    if number >= 10_000_000:
        return f"₹{number / 10_000_000:,.2f} Cr"
    elif number >= 100_000:
        return f"₹{number / 100_000:,.2f} Lakh"
    return f"₹{number:,.2f}"


def format_usd(number):
    if number >= 1_000_000_000:
        return f"${number / 1_000_000_000:,.2f} Billion"
    elif number >= 1_000_000:
        return f"${number / 1_000_000:,.2f} Million"
    elif number >= 1_000:
        return f"${number / 1_000:,.2f}K"
    return f"${number:,.2f}"


# -------------------------------------------------
# 5. SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("## 🎬 IMDb Movie")
    st.markdown("## Gross Predictor")
    st.caption("A simple machine learning tool for box office estimation")

    st.divider()

    st.markdown("### 🏠 Navigation")
    st.radio(
        "Choose Section",
        ["Home", "Prediction", "About"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 💱 Currency Settings")
    currency = st.radio(
        "Display Currency",
        ["USD ($)", "INR (₹)"]
    )

    usd_to_inr = 83.0

    st.markdown("""
    <div class="sidebar-card">
        <h4>ℹ️ About This Project</h4>
        <p>
        This application uses a trained Artificial Neural Network
        to estimate worldwide movie gross based on historical IMDb data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <h4>📌 Main Predictors</h4>
        <p>
        • IMDb Rating<br>
        • Metascore<br>
        • Movie Duration<br>
        • Number of Votes
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <h4>🧠 How It Works</h4>
        <p>
        1. Enter movie details<br>
        2. Input values are normalized<br>
        3. The ANN model generates an estimate<br>
        4. Result is displayed with currency formatting
        </p>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------
# 6. HERO HEADER
# -------------------------------------------------
st.markdown("""
<div class="hero-card">
    <div class="hero-badge">● Machine Learning Prediction Tool</div>
    <h1>🎬 IMDb Movie Gross Predictor</h1>
    <p>
        Estimate worldwide box office earnings using historical IMDb data
        and a trained Artificial Neural Network model.
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# 7. QUICK PRESETS
# -------------------------------------------------
st.markdown("""
<div class="section-title">⚡ Quick Load Presets</div>
<div class="section-subtitle">
Choose a sample movie profile or enter your own values.
</div>
""", unsafe_allow_html=True)

preset = st.selectbox(
    "Movie Scenario",
    [
        "Custom Input",
        "Blockbuster Action",
        "Critically Acclaimed Drama",
        "Indie Low-Budget"
    ]
)

def_rating = 8.0
def_meta = 75
def_dur = 120
def_votes = 100000

if preset == "Blockbuster Action":
    def_rating = 8.8
    def_meta = 74
    def_dur = 148
    def_votes = 2400000

elif preset == "Critically Acclaimed Drama":
    def_rating = 8.5
    def_meta = 92
    def_dur = 130
    def_votes = 500000

elif preset == "Indie Low-Budget":
    def_rating = 6.8
    def_meta = 60
    def_dur = 95
    def_votes = 25000


# -------------------------------------------------
# 8. INPUT AND RESULT LAYOUT
# -------------------------------------------------
input_col, result_col = st.columns([1.05, 0.95], gap="large")


# -------------------------------------------------
# 9. MOVIE INPUTS
# -------------------------------------------------
with input_col:
    st.markdown("""
    <div class="section-title">🎞️ Movie Details</div>
    <div class="section-subtitle">
    Enter the basic movie information below.
    </div>
    """, unsafe_allow_html=True)

    imdb_rating = st.slider(
        "⭐ IMDb Rating",
        min_value=1.0,
        max_value=10.0,
        value=float(def_rating),
        step=0.1
    )

    metascore = st.slider(
        "🎯 Metascore",
        min_value=0,
        max_value=100,
        value=int(def_meta),
        step=1
    )

    duration = st.number_input(
        "⏱️ Duration (Minutes)",
        min_value=30,
        max_value=300,
        value=int(def_dur),
        step=5
    )

    votes = st.number_input(
        "🗳️ Votes Count",
        min_value=1000,
        max_value=3000000,
        value=int(def_votes),
        step=5000
    )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button("📊 Predict Box Office Gross")


# -------------------------------------------------
# 10. RESULT SECTION
# -------------------------------------------------
with result_col:
    st.markdown("""
    <div class="section-title">📈 Prediction Result</div>
    <div class="section-subtitle">
    Your estimated worldwide box office collection will appear here.
    </div>
    """, unsafe_allow_html=True)

    if predict_clicked:
        raw_input_data = np.array([
            [imdb_rating, metascore, duration, votes]
        ])

        scaled_data = scaler.transform(raw_input_data)

        prediction = model.predict(scaled_data)

        final_gross = float(np.asarray(prediction).reshape(-1)[0])
        final_gross = max(0, final_gross)

        if currency == "INR (₹)":
            display_value = format_inr(final_gross * usd_to_inr)
            currency_name = "Indian Rupees"
        else:
            display_value = format_usd(final_gross)
            currency_name = "US Dollars"

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">ESTIMATED WORLDWIDE GROSS</div>
            <div class="result-value">{display_value}</div>
            <div class="result-note">
                Estimated value in {currency_name}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📝 Movie Input Summary")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.metric("IMDb Rating", f"{imdb_rating:.1f}/10")
            st.metric("Duration", f"{duration} minutes")

        with summary_col2:
            st.metric("Metascore", f"{metascore}/100")
            st.metric("Votes", f"{votes:,}")

        st.markdown("""
        <div class="info-box">
        ℹ️ This is an estimated prediction based on historical IMDb data.
        Actual box office performance may differ because of marketing,
        audience interest, release timing, competition and other factors.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-card">
            <div class="result-label">PREDICTION PREVIEW</div>
            <div class="result-value">—</div>
            <div class="result-note">
                Enter the movie details and click the prediction button.
            </div>
        </div>
        """, unsafe_allow_html=True)


# -------------------------------------------------
# 11. CHART AFTER PREDICTION
# -------------------------------------------------
if predict_clicked:
    st.markdown("---")

    st.markdown("""
    <div class="section-title">📊 Input Feature Overview</div>
    <div class="section-subtitle">
    A normalized view of the values supplied to the model.
    </div>
    """, unsafe_allow_html=True)

    categories = [
        "IMDb Rating",
        "Metascore",
        "Duration",
        "Votes"
    ]

    norm_values = [
        imdb_rating * 10,
        metascore,
        (duration / 300) * 100,
        min((votes / 1_000_000) * 100, 100)
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=norm_values,
                marker_color="#3978c7",
                text=[f"{value:.1f}" for value in norm_values],
                textposition="outside"
            )
        ]
    )

    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial",
            color="#34435b"
        ),
        yaxis=dict(
            title="Normalized Scale (0–100)",
            range=[0, 115],
            gridcolor="#e2e8f0"
        ),
        xaxis=dict(
            gridcolor="#e2e8f0"
        ),
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    if final_gross > 100_000_000:
        st.success(
            "🎉 This movie profile has a high estimated worldwide gross."
        )


# -------------------------------------------------
# 12. FOOTER
# -------------------------------------------------
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit and Machine Learning |
    <a href="https://github.com/AishAftab098" target="_blank">
        GitHub
    </a>
    &nbsp; • &nbsp;
    <a href="https://www.linkedin.com/in/aish-aftab" target="_blank">
        LinkedIn
    </a>
    <br>
    IMDb Movie Gross Predictor — Academic Machine Learning Project
</div>
""", unsafe_allow_html=True)