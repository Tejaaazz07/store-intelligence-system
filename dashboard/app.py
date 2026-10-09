import streamlit as st
import requests

st.set_page_config(
    page_title="Purplle Store Intelligence",
    page_icon="🛍️",
    layout="wide"
)

# -----------------------------------
# CSS
# -----------------------------------

st.markdown("""
<style>

.stApp{
    background:#050816;
}

/* remove streamlit padding */
.block-container{
    padding-top:1rem;
    padding-bottom:1rem;
}

/* Header */
.hero{
    background:linear-gradient(
        90deg,
        #6D28D9,
        #8B5CF6,
        #A855F7
    );

    border-radius:25px;
    padding:35px 40px;
    margin-bottom:25px;

    width:100%;
    box-sizing:border-box;
}


.hero-title{
    color:white;
    font-size:48px;
    font-weight:800;
}

.hero-sub{
    color:white;
    font-size:20px;
}

/* KPI cards */

.card{
    background:#140f33;
    border:1px solid #7C3AED;
    border-radius:20px;
    padding:18px;
    text-align:center;
    min-height:100px;
}

.card-icon{
    font-size:28px;
}

.card-title{
    color:white;
    font-size:18px;
    font-weight:600;
}

.card-value{
    color:white;
    font-size:36px;
    font-weight:800;
}

/* status cards */

.status-title{
    color:white;
    font-size:36px;
    font-weight:700;
}

.red-box{
    background:#3a1016;
    border:1px solid #ef4444;
    border-radius:20px;
    padding:25px;
    min-height:180px;
}

.green-box{
    background:#0d331d;
    border:1px solid #22c55e;
    border-radius:20px;
    padding:25px;
    min-height:180px;
}

.box-heading{
    color:white;
    font-size:22px;
    font-weight:700;
}

.box-text{
    color:white;
    font-size:16px;
}

/* health cards */

.health{
    background:#111827;
    border:1px solid #7C3AED;
    border-radius:18px;
    padding:20px;
    text-align:center;
}

.health-title{
    color:white;
    font-size:18px;
    font-weight:700;
}

.health-ok{
    color:#22c55e;
    font-size:18px;
    font-weight:700;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# API DATA
# -----------------------------------

try:

    metrics = requests.get(
        "http://127.0.0.1:8000/metrics"
    ).json()

    crowding = requests.get(
        "http://127.0.0.1:8000/crowding"
    ).json()

    anomaly = requests.get(
        "http://127.0.0.1:8000/anomaly"
    ).json()

except:
    st.error("FastAPI backend is not running")
    st.stop()

# -----------------------------------
# HEADER
# -----------------------------------

st.markdown("""
<div class="hero">
<div class="hero-title">
🛍️ Purplle Store Intelligence Dashboard
</div>

<div class="hero-sub">
Retail Analytics & CCTV Intelligence Platform
</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# KPI ROW
# -----------------------------------

c1,c2,c3,c4,c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="card">
    <div class="card-icon">📊</div>
    <div class="card-title">EVENTS</div>
    <div class="card-value">{metrics['total_events']}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
    <div class="card-icon">🚪</div>
    <div class="card-title">ENTRIES</div>
    <div class="card-value">{metrics['entries']}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
    <div class="card-icon">🏃</div>
    <div class="card-title">EXITS</div>
    <div class="card-value">{metrics['exits']}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="card">
    <div class="card-icon">👥</div>
    <div class="card-title">VISITORS</div>
    <div class="card-value">{crowding['current_people']}</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown("""
    <div class="card">
    <div class="card-icon">💜</div>
    <div class="card-title">HEALTH</div>
    <div class="card-value">OK</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -----------------------------------
# STATUS SECTION
# -----------------------------------

left,right = st.columns(2)

with left:

    st.markdown(
        "<div class='status-title'>👥 Crowding Status</div>",
        unsafe_allow_html=True
    )

    if crowding["crowding_alert"]:

        st.markdown(f"""
        <div class="red-box">
        <div class="box-heading">🚨 CROWDING ALERT</div>
        <br>
        <div class="box-text">
        Current Visitors : {crowding['current_people']}
        </div>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="green-box">
        <div class="box-heading">✅ NORMAL</div>
        <br>
        <div class="box-text">
        Current Visitors : {crowding['current_people']}
        </div>
        </div>
        """, unsafe_allow_html=True)

with right:

    st.markdown(
        "<div class='status-title'>🛡️ Anomaly Detection</div>",
        unsafe_allow_html=True
    )

    if anomaly["anomaly"]:

        st.markdown(f"""
        <div class="red-box">
        <div class="box-heading">🚨 ANOMALY DETECTED</div>
        <br>
        <div class="box-text">
        {anomaly['reason']}
        </div>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="green-box">
        <div class="box-heading">✅ NORMAL ACTIVITY</div>
        <br>
        <div class="box-text">
        {anomaly['reason']}
        </div>
        </div>
        """, unsafe_allow_html=True)

st.write("")
st.write("")

# -----------------------------------
# SYSTEM HEALTH
# -----------------------------------

st.markdown(
    "<div class='status-title'>⚙️ System Status</div>",
    unsafe_allow_html=True
)

h1,h2,h3 = st.columns(3)

with h1:
    st.markdown("""
    <div class="health">
    <div class="health-title">🖥️ FastAPI Backend</div>
    <br>
    <div class="health-ok">Connected</div>
    </div>
    """, unsafe_allow_html=True)

with h2:
    st.markdown("""
    <div class="health">
    <div class="health-title">🗄️ SQLite Database</div>
    <br>
    <div class="health-ok">Connected</div>
    </div>
    """, unsafe_allow_html=True)

with h3:
    st.markdown("""
    <div class="health">
    <div class="health-title">📈 Analytics Engine</div>
    <br>
    <div class="health-ok">Running</div>
    </div>
    """, unsafe_allow_html=True)