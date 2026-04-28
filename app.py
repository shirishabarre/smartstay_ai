import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
import pydeck as pdk
import json
import re   # ✅ NEW (for JSON extraction)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ================= CONFIG =================
st.set_page_config(page_title="SmartStay AI", layout="wide")

# ================= LOAD ENV =================
load_dotenv()

# ================= GROQ =================
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ API Key not found")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ================= LOGIN =================
USERS = {
    "user1": "1234",
    "admin": "admin123"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.markdown('<div class="main-title">🔐 SmartStay Login</div>', unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid Credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ================= STYLING =================
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #020617, #0f172a, #1e293b);
    color: white;
}

.main-title {
    font-size: 50px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-title { text-align: center; color: #cbd5f5; margin-bottom: 30px; }

.glass-card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
}

.result-box {
    background: rgba(255,255,255,0.07);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    border-left: 6px solid #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="main-title">🏠 SmartStay AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Find Best Hostels, PGs, Flats & Villas with AI</div>', unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_resource
def load_vector_db():
    loader = PyPDFLoader("data/SmartStay_Full_Dataset_Updated.pdf")
    documents = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory="db"
    )

    return vectordb.as_retriever(search_kwargs={"k": 5})

retriever = load_vector_db()

# ================= LOCATION COORDS =================
location_coords = {
    "JNTU": [17.4948, 78.3917],
    "KPHB": [17.4945, 78.3995],
    "Kukatpally": [17.4849, 78.4138],
    "Miyapur": [17.4940, 78.3500],
    "Ameerpet": [17.4375, 78.4483],
    "Hitech City": [17.4485, 78.3915],
    "Gachibowli": [17.4401, 78.3489],
    "Madhapur": [17.4483, 78.3915],
    "SR Nagar": [17.4550, 78.4440],
    "Bachupally": [17.5440, 78.3800]
}

# ================= INPUT UI =================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("📍 Location", list(location_coords.keys()))
    accommodation = st.selectbox("🏡 Accommodation", ["Hostel", "PG", "Flat", "Villa"])

with col2:
    gender = st.radio("Gender / Preference", ["Men", "Women", "Bachelor", "Couple"])
    budget = st.slider("💰 Budget", 5000, 50000, 15000)
    sharing = st.selectbox("🛏 Sharing", ["Single", "Double", "Triple"])

st.markdown('</div>', unsafe_allow_html=True)

# ================= QUERY =================
query = f"""
Location: {location}
Type: {accommodation}
Gender: {gender}
Budget: {budget}
Sharing: {sharing}
"""

# ================= JSON FIX FUNCTION =================
def extract_json(text):
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        return None
    return None

# ================= RAG =================
def generate_answer(user_query):
    docs = retriever.invoke(user_query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Return ONLY JSON.

Format:
[
  {{
    "name": "",
    "price": "",
    "location": "",
    "sharing": "",
    "description": ""
  }}
]

User Query:
{user_query}

Data:
{context}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

# ================= SEARCH =================
if st.button("🔍 Search"):
    with st.spinner("Finding best stays..."):

        raw_output = generate_answer(query)
        stays = extract_json(raw_output)

        st.markdown("## ✅ Recommendations")

        if not stays:
            st.error("⚠️ Could not parse JSON")
            st.write(raw_output)
        else:
            for stay in stays:

                st.markdown(f"""
                <div class="result-box">
                <h3>{stay['name']}</h3>
                💰 Price: ₹{stay['price']} <br>
                📍 Location: {stay['location']} <br>
                🛏 Sharing: {stay['sharing']} <br><br>
                {stay['description']}
                </div>
                """, unsafe_allow_html=True)

                # ===== MAP =====
                loc = stay.get("location", location)
                lat, lon = location_coords.get(loc, location_coords[location])

                map_data = pd.DataFrame({
                    "lat": [lat],
                    "lon": [lon]
                })

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_data,
                    get_position='[lon, lat]',
                    get_radius=120
                )

                view_state = pdk.ViewState(
                    latitude=lat,
                    longitude=lon,
                    zoom=14
                )

                deck = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip={"text": stay["name"]}
                )

                st.pydeck_chart(deck)