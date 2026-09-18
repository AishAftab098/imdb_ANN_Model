import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="IMDb Movie Gross Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #F5F7FB;
    color: #172033;
}

.main .block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color: #EEF3F9;
    border-right: 1px solid #D8E0EA;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #17365D !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
    color: #334155 !important;
}


/* NORMAL TEXT */

h1, h2, h3, h4 {
    color: #172033 !important;
}

p {
    color: #526174;
}


/* HERO */

.hero {
    background: linear-gradient(135deg, #173F67, #2F6690);
    border-radius: 18px;
    padding: 34px 40px;
    margin-bottom: 30px;
    box-shadow: 0 8px 24px rgba(23, 63, 103, 0.15);
}

.hero-tag {
    display: inline-block;
    background-color: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    padding: 7px 14px;
    border-radius: 20px;
    font-size: 13px;
    margin-bottom: 15px;
}

.hero-title {
    color: white !important;
    font-size: 40px;
    font-weight: 700;
    margin: 0;
}

.hero-subtitle {
    color: #E1EDF7 !important;
    font-size: 16px;
    margin-top: 12px;
}


/* SECTION TITLES */

.section-title {
    color: #17365D;
    font-size: 25px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 5px;
}

.section-description {
    color: #68778A;
    font-size: 14px;
    margin-bottom: 18px;
}


/* INPUT CARD */

.input-card {
    background-color: white;
    border: 1px solid #DDE4EC;
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 3px 12px rgba(30,50,70,0.05);
}


/* RESULT CARD */

.result-card {
    background-color: #F0FBF6;
    border: 1px solid #B8E4CF;
    border-radius: 15px;
    padding: 28px 20px;
    text-align: center;
    min-height: 170px;
}

.result-label {
    color: #34745B !important;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.result-value {
    color: #16865B !important;
    font-size: 34px;
    font-weight: 700;
    margin: 12px 0;
}

.result-note {
    color: #718096 !important;
    font-size: 13px;
}


/* BUTTON */

.stButton > button {
    width: 100%;
    background-color: #173F67;
    color: white !important;
    border: none;
    border-radius: 9px;
    padding: 12px 20px;
    font-size: 16px;
    font-weight: 600;
    min-height: 48px;
}

.stButton > button:hover {
    background-color: #245985;
}


/* ABOUT CARD */

.about-card {
    background-color: white;
    border: 1px solid #D8E0EA;
    border-radius: 13px;
    padding: 18px;
    margin-top: 20px;
}

.about-title {
    color: #17365D !important;
    font-size: 16px;
    font-weight: 700;
}

.about-text {
    color: #64748B !important;
    font-size: 13px;
    line-height: 1.6;
}


/* FOOTER */

.footer {
    text-align: center;
    color: #7B8794 !important;
    font-size: 13px;
    padding-top: 30px;
    margin-top: 45px;
    border-top: 1px solid #DCE2E9;
}

.footer a {
    color: #173F67 !important;
    text-decoration: none;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_artifacts():

    with open("imdb_ann_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


try:
    model, scaler = load_artifacts()

except Exception as e:
    st.error(f"Model loading error: {e}")
        
    st.stop()


# =========================================================
# INR FORMAT
# =========================================================

def format_inr(number):

    if number >= 10_000_000:
        return f"₹{number / 10_000_000:,.2f} Cr"

    elif number >= 100_000:
        return f"₹{number / 100_000:,.2f} Lakh"

    else:
        return f"₹{number:,.2f}"


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "<h1>🎬 IMDb Movie<br>Gross Predictor</h1>",
        unsafe_allow_html=True
    )

    st.write(
        "A simple machine learning tool for "
        "worldwide box office estimation."
    )

    st.markdown("---")

    st.subheader("🧭 Navigation")

    st.radio(
        "Navigation",
        ["🏠 Home", "📈 Prediction", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.subheader("💱 Currency Settings")

    currency = st.radio(
        "Display Currency",
        ["USD ($)", "INR (₹)"]
    )

    usd_to_inr = 83.0

    st.markdown(
        """
<div class="about-card">
<div class="about-title">ℹ️ About This Project</div>
<p class="about-text">
This application uses a trained Artificial Neural Network
to estimate worldwide movie gross based on selected IMDb
movie characteristics.
</p>
</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    """
<div class="hero">
<div class="hero-tag">● Machine Learning Prediction Tool</div>
<div class="hero-title">🎬 IMDb Movie Gross Predictor</div>
<div class="hero-subtitle">
Estimate worldwide box office earnings using historical IMDb
data and a trained Artificial Neural Network model.
</div>
</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# QUICK PRESETS
# =========================================================

st.markdown(
    '<div class="section-title">⚡ Quick Load Presets</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Choose a sample movie profile or enter your own values.'
    '</div>',
    unsafe_allow_html=True
)


preset = st.selectbox(
    "Movie Scenario",
    [
        "Custom Input",
        "Blockbuster Action",
        "Critically Acclaimed Drama",
        "Indie Low-Budget"
    ]
)


# Default values

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


# =========================================================
# MOVIE DETAILS
# =========================================================

st.markdown(
    '<div class="section-title">🎞️ Movie Details</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Enter the basic movie information below.'
    '</div>',
    unsafe_allow_html=True
)

left_col, right_col = st.columns(
    [1.35, 0.9],
    gap="large"
)

# =========================================================
# LEFT COLUMN - MOVIE FEATURES
# =========================================================

with left_col:

    st.markdown("### 🎬 Movie Features")

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


# =========================================================
# RIGHT COLUMN - PREDICTION PREVIEW
# =========================================================

with right_col:

    st.markdown("### 📈 Prediction Result")

    st.markdown(
        """
        <div class="result-card">
            <div class="result-label">PREDICTION PREVIEW</div>
            <div class="result-value">—</div>
            <div class="result-note">
                Enter the movie details and click the prediction button below.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

button_col1, button_col2, button_col3 = st.columns(
    [1, 1.4, 1]
)

with button_col2:

    predict_button = st.button(
        "🚀 Predict Box Office Gross",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    input_data = np.array([
        [
            imdb_rating,
            metascore,
            duration,
            votes
        ]
    ])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    final_gross = float(
        np.asarray(prediction).flatten()[0]
    )

    final_gross = max(0, final_gross)

    # =====================================================
    # CURRENCY
    # =====================================================

    if currency == "INR (₹)":

        display_value = format_inr(
            final_gross * usd_to_inr
        )

        currency_note = "Approximate INR conversion"

    else:

        display_value = f"${final_gross:,.2f}"

        currency_note = "Estimated worldwide gross"


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">
                ESTIMATED WORLDWIDE BOX OFFICE
            </div>

            <div class="result-value">
                {display_value}
            </div>

            <div class="result-note">
                {currency_note}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    input_data = np.array([
        [
            imdb_rating,
            metascore,
            duration,
            votes
        ]
    ])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    final_gross = float(
        np.asarray(prediction).flatten()[0]
    )

    final_gross = max(0, final_gross)


    # Currency

    if currency == "INR (₹)":

        display_value = format_inr(
            final_gross * usd_to_inr
        )

        currency_note = "Approximate INR conversion"

    else:

        display_value = f"${final_gross:,.2f}"

        currency_note = "Estimated worldwide gross"


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        f"""
<div class="result-card">
<div class="result-label">
ESTIMATED WORLDWIDE BOX OFFICE
</div>
<div class="result-value">
{display_value}
</div>
<div class="result-note">
{currency_note}
</div>
</div>
""",
        unsafe_allow_html=True
    )


    # =====================================================
    # CHART
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">'
        '📊 Movie Feature Overview'
        '</div>',
        unsafe_allow_html=True
    )

    categories = [
        "IMDb Rating",
        "Metascore",
        "Duration",
        "Votes"
    ]

    normalized_values = [
        imdb_rating * 10,
        metascore,
        (duration / 300) * 100,
        min((votes / 3000000) * 100, 100)
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=categories,
            y=normalized_values,
            marker_color="#2B6CB0",
            text=[
                f"{imdb_rating:.1f}",
                f"{metascore}",
                f"{duration} min",
                f"{votes:,}"
            ],
            textposition="outside"
        )
    )

    fig.update_layout(
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font=dict(
            color="#334155",
            size=13
        ),
        yaxis=dict(
            title="Relative Scale (0–100)",
            range=[0, 110],
            gridcolor="#E2E8F0",
            zeroline=False
        ),
        xaxis=dict(
            gridcolor="#FFFFFF"
        ),
        margin=dict(
            l=40,
            r=30,
            t=35,
            b=40
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # =====================================================
    # MESSAGE
    # =====================================================

    if final_gross >= 100_000_000:

        st.success(
            "🎉 The model estimates a worldwide gross "
            "above $100 million."
        )

    elif final_gross >= 50_000_000:

        st.info(
            "📈 The model estimates a worldwide gross "
            "above $50 million."
        )

    else:

        st.info(
            "📊 Prediction generated successfully using "
            "the trained ANN model."
        )


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
### 🧠 Model

**Artificial Neural Network (ANN)**

The trained model estimates worldwide
box office gross from movie features.
"""
    )


with col2:

    st.markdown(
        """
### 📌 Input Features

- IMDb Rating
- Metascore
- Duration
- Votes Count
"""
    )


with col3:

    st.markdown(
        """
### 🛠️ Technologies

- Python
- Streamlit
- NumPy
- Plotly
- Machine Learning
"""
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">
IMDb Movie Gross Predictor
&nbsp; • &nbsp;
Built with Streamlit and Machine Learning
<br><br>
<a href="https://github.com/AishAftab098"
target="_blank">GitHub</a>
&nbsp; • &nbsp;
<a href="https://www.linkedin.com/in/aish-aftab"
target="_blank">LinkedIn</a>
</div>
""",
    unsafe_allow_html=True
)