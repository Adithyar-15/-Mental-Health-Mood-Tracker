import streamlit as st
import pickle
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load model and vectorizer
with open('emotion_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Emotion data
emotion_emoji = {
    'joy': '😊', 'sadness': '😢',
    'anger': '😠', 'fear': '😨',
    'love': '❤️', 'surprise': '😲'
}
emotion_color = {
    'joy': '#F4A261',
    'sadness': '#C9407A',
    'anger': '#E8A0BF',
    'fear': '#9B72CF',
    'love': '#FF85A1',
    'surprise': '#FFB347'
}
emotion_bg = {
    'joy': '#FFF3E0', 'sadness': '#FFF0F5',
    'anger': '#FFF5F8', 'fear': '#F5EEFF',
    'love': '#FFF0F5', 'surprise': '#FFF8EE'
}

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Predict
def predict_emotion(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized).max() * 100
    return prediction, probability

# Database
def init_db():
    conn = sqlite3.connect('mood_journal.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS entries
                 (date TEXT, text TEXT, emotion TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

def save_entry(text, emotion, confidence):
    conn = sqlite3.connect('mood_journal.db')
    c = conn.cursor()
    date = datetime.now().strftime('%Y-%m-%d %H:%M')
    c.execute('INSERT INTO entries VALUES (?,?,?,?)',
              (date, text, emotion, confidence))
    conn.commit()
    conn.close()

def get_entries():
    conn = sqlite3.connect('mood_journal.db')
    df = pd.read_sql('SELECT * FROM entries', conn)
    conn.close()
    return df

init_db()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="MoodTracker AI",
    page_icon="🌸",
    layout="centered"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Poppins:wght@300;400;500;600&display=swap');

    .stApp {
        background-color: #FDF6F0;
        font-family: 'Poppins', sans-serif;
    }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #FFF0F3 0%, #FDF6F0 100%);
        border-radius: 30px;
        padding: 50px 40px 35px 40px;
        text-align: center;
        margin-bottom: 30px;
        border: 1px solid #FFE4EC;
        box-shadow: 0 10px 40px rgba(232,160,191,0.15);
    }
    .hero h1 {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        color: #2D2D2D;
        line-height: 1.2;
        margin: 10px 0;
    }
    .hero h1 span {
        color: #C9407A;
    }
    .hero p {
        color: #888;
        font-size: 1rem;
        font-weight: 300;
        margin-top: 15px;
        line-height: 1.7;
    }

    /* Mood bubbles */
    .mood-strip {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin: 25px 0 5px 0;
        flex-wrap: wrap;
    }
    .mood-bubble {
        display: flex;
        flex-direction: column;
        align-items: center;
        background: white;
        border: 1.5px solid #FFE4EC;
        border-radius: 20px;
        padding: 12px 16px;
        font-size: 1.8rem;
        cursor: default;
        animation: float 3s ease-in-out infinite;
        box-shadow: 0 4px 15px rgba(201,64,122,0.08);
        transition: transform 0.3s;
    }
    .mood-bubble span {
        font-size: 0.7rem;
        color: #C9407A;
        font-weight: 500;
        margin-top: 5px;
        font-family: 'Poppins', sans-serif;
    }
    .mood-bubble:hover {
        transform: translateY(-8px) scale(1.1);
        box-shadow: 0 10px 25px rgba(201,64,122,0.15);
        border-color: #C9407A;
    }
    .mood-bubble.active {
        background: linear-gradient(135deg, #FFF0F3, #FFE4EC);
        border-color: #C9407A;
        transform: translateY(-5px);
    }
    .mood-bubble:nth-child(1) { animation-delay: 0s; }
    .mood-bubble:nth-child(2) { animation-delay: 0.3s; }
    .mood-bubble:nth-child(3) { animation-delay: 0.6s; }
    .mood-bubble:nth-child(4) { animation-delay: 0.9s; }
    .mood-bubble:nth-child(5) { animation-delay: 1.2s; }
    .mood-bubble:nth-child(6) { animation-delay: 1.5s; }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }

    /* Input card */
    .input-card {
        background: white;
        border-radius: 25px;
        padding: 35px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.06);
        border: 1px solid #FFE4EC;
        margin-bottom: 25px;
    }
    .input-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2D2D2D;
        margin-bottom: 5px;
    }
    .input-sublabel {
        font-size: 0.85rem;
        color: #aaa;
        margin-bottom: 15px;
    }

    /* Text area */
    .stTextArea textarea {
        border-radius: 15px !important;
        border: 1.5px solid #FFE4EC !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.95rem !important;
        color: #2D2D2D !important;
        padding: 15px !important;
        background: #FFFAF9 !important;
    }
    .stTextArea textarea:focus {
        border: 1.5px solid #C9407A !important;
        box-shadow: 0 0 15px rgba(201,64,122,0.1) !important;
    }

    /* Button */
    .stButton button {
        background: linear-gradient(135deg, #E8A0BF, #C9407A) !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 14px 40px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        border: none !important;
        width: 100% !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 8px 25px rgba(201,64,122,0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px rgba(201,64,122,0.4) !important;
    }

    /* Result card */
    .result-card {
        border-radius: 25px;
        padding: 35px;
        text-align: center;
        margin: 20px 0;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0 8px 30px rgba(0,0,0,0.06);
        animation: fadeIn 0.5s ease;
    }
    .result-emoji {
        font-size: 4.5rem;
        margin-bottom: 15px;
        display: block;
        animation: bounceIn 0.6s ease;
    }
    .result-emotion {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 5px 0;
    }
    .result-confidence {
        font-size: 0.9rem;
        color: #aaa;
        margin-top: 8px;
        font-weight: 300;
    }
    .result-saved {
        margin-top: 15px;
        font-size: 0.85rem;
        color: #52B788;
        font-weight: 500;
    }

    /* Stat cards */
    .stat-card {
        background: white;
        border-radius: 20px;
        padding: 25px 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        border: 1px solid #FFE4EC;
    }
    .stat-number {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 700;
        color: #C9407A;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #aaa;
        margin-top: 5px;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Section title */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        color: #2D2D2D;
        text-align: center;
        margin: 35px 0 20px 0;
    }
    .section-title span { color: #C9407A; }

    /* Chart card */
    .chart-card {
        background: white;
        border-radius: 25px;
        padding: 25px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.06);
        border: 1px solid #FFE4EC;
        margin-bottom: 20px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #ccc;
        padding: 30px;
        font-size: 0.82rem;
        font-weight: 300;
    }
    .footer span { color: #E8A0BF; }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes bounceIn {
        0% { transform: scale(0.5); opacity: 0; }
        70% { transform: scale(1.2); }
        100% { transform: scale(1); opacity: 1; }
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- HERO ---
st.markdown("""
<div class="hero">
    <h1>Understand Your<br><span>Inner Emotions</span></h1>
    <p>Write how you feel — our AI reads between the lines<br>
    and helps you understand your emotional patterns</p>
    <div class="mood-strip">
        <div class="mood-bubble">😊<span>Joy</span></div>
        <div class="mood-bubble active">❤️<span>Love</span></div>
        <div class="mood-bubble">😢<span>Sad</span></div>
        <div class="mood-bubble">😨<span>Fear</span></div>
        <div class="mood-bubble">😲<span>Surprise</span></div>
        <div class="mood-bubble">😠<span>Anger</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- INPUT CARD ---
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="input-label">📝 Today\'s Journal Entry</div>',
            unsafe_allow_html=True)
st.markdown('<div class="input-sublabel">Write freely — your thoughts are safe here</div>',
            unsafe_allow_html=True)

journal = st.text_area(
    "",
    placeholder="I feel really happy and grateful today...",
    height=160,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze = st.button("🌸 Analyze My Mood")

st.markdown('</div>', unsafe_allow_html=True)

# --- RESULT ---
if analyze:
    if journal.strip() == "":
        st.warning("💬 Please write something first!")
    else:
        emotion, confidence = predict_emotion(journal)
        emoji = emotion_emoji[emotion]
        color = emotion_color[emotion]
        bg = emotion_bg[emotion]

        st.markdown(f"""
        <div class="result-card" style="background:{bg};">
            <span class="result-emoji">{emoji}</span>
            <div class="result-emotion" style="color:{color};">
                {emotion.title()}
            </div>
            <div class="result-confidence">
                Confidence Score: {confidence:.1f}%
            </div>
            <div class="result-saved">✓ Saved to your journal</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence))
        save_entry(journal, emotion, confidence)

# --- DASHBOARD ---
df = get_entries()

if not df.empty:
    st.markdown("""
    <div class="section-title">Your Mood <span>Dashboard</span></div>
    """, unsafe_allow_html=True)

    total = len(df)
    top_emotion = df['emotion'].mode()[0]
    avg_conf = df['confidence'].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{total}</div>
            <div class="stat-label">Total Entries</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{emotion_emoji[top_emotion]}</div>
            <div class="stat-label">Top Emotion</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{avg_conf:.0f}%</div>
            <div class="stat-label">Avg Confidence</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar chart
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig = px.bar(df, x='date', y='confidence',
                 color='emotion',
                 title='Mood Trend Over Time',
                 color_discrete_map=emotion_color)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family='Poppins',
        font_color='#2D2D2D',
        title_font_size=15,
        title_font_color='#2D2D2D',
        showlegend=True,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True,
                   gridcolor='#FFE4EC')
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Pie chart
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    pie = px.pie(df, names='emotion',
                 title='Emotion Breakdown',
                 color='emotion',
                 color_discrete_map=emotion_color,
                 hole=0.45)
    pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_family='Poppins',
        font_color='#2D2D2D',
        title_font_size=15,
        title_font_color='#2D2D2D'
    )
    st.plotly_chart(pie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Recent entries
    st.markdown("""
    <div class="section-title">Recent <span>Entries</span></div>
    """, unsafe_allow_html=True)
    st.dataframe(
        df.tail(5)[['date', 'emotion', 'confidence']],
        use_container_width=True
    )

else:
    st.markdown("""
    <div style="text-align:center; padding:30px; color:#C9407A; 
                font-family:'Poppins',sans-serif; opacity:0.7;">
        No entries yet — analyze your first mood above! 👆
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div class="footer">
    Built with <span>♥</span> using Python · Machine Learning · NLP · Streamlit
</div>
""", unsafe_allow_html=True)