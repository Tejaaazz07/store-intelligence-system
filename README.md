# Purplle Store Intelligence System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-orange)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
# 🛍️ Purplle Store Intelligence System

AI-powered retail analytics platform built for the **Purplle Store Intelligence Challenge 2026**.

The system processes CCTV footage, tracks customer movement, generates store events, computes operational KPIs, and provides business insights through REST APIs and an interactive dashboard.

---

## 🚀 Features

### Computer Vision Pipeline

- YOLOv8 Person Detection
- Multi-Person Tracking
- Entry Detection
- Exit Detection
- Zone Entry Tracking
- Zone Exit Tracking
- Visitor Session Tracking

### Analytics Engine

- Occupancy Analytics
- Peak Occupancy Tracking
- Dwell Time Analytics
- Funnel Analytics
- KPI Dashboard
- Crowding Detection
- Anomaly Detection
- Queue Completion Analytics
- Queue Abandonment Analytics
- POS Transaction Analytics

### Platform

- FastAPI Backend
- Interactive Streamlit Dashboard
- SQLite Event Storage
- Swagger API Documentation
- Docker Deployment
- Health Monitoring Endpoint

---

## 🏗️ System Architecture

```text
CCTV Video
     ↓
YOLOv8 Detection
     ↓
Person Tracking
     ↓
Event Generation
     ↓
Zone / Queue / POS Analytics
     ↓
SQLite Database
     ↓
Analytics Engine
     ↓
FastAPI APIs
     ↓
Streamlit Dashboard
```

---

## 📂 Project Structure

```text
store-intelligence-system/
├── app/
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── analytics.py
│   ├── occupancy.py
│   ├── dwell_time.py
│   ├── crowding.py
│   ├── anomaly.py
│   ├── kpi.py
│   └── pos.py

├── dashboard/
│   └── app.py

├── pipeline/
│   ├── detect_people.py
│   ├── track_people.py
│   ├── event_generator.py
│   ├── session_tracker.py
│   ├── zone_tracker.py
│   └── zones.py

├── sample_data/
│   ├── pos_transaction.json
│   ├── sample_datastore_video.mp4
│   └── output.mp4

├── docs/
│   ├── architecture.md
│   ├── dashboard.png
│   ├── swagger.png
│   ├── camera_output_1.png
│   ├── camera_output_2.png
│   └── PROGRESS.md

├── tests/

├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── DESIGN.md
├── CHOICES.md
├── store.db
└── yolov8n.pt
```

---

## 📊 Dashboard Features

The Streamlit dashboard provides:

- Total Events
- Entry Count
- Exit Count
- Active Visitors
- System Health Status
- Crowding Alerts
- Anomaly Monitoring
- KPI Overview
- Queue Monitoring
- Zone Activity Insights
- POS Performance Metrics

---

# 📸 Screenshots

## Dashboard

Interactive retail analytics dashboard.

![Dashboard](docs/dashboard.png)

---

## Swagger API Documentation

FastAPI auto-generated API documentation.

![Swagger](docs/swagger.png)

---

## Detection Pipeline Output

YOLOv8 detecting and tracking customers inside the retail store.

![Camera Output 1](docs/camera_output_1.png)

---

## Visitor Tracking Example 2

Multi-person tracking with unique visitor IDs.

![Camera Output 2](docs/camera_output_2.png)

---

# 🐳 Running With Docker

Build and start the application:

```bash
docker compose up --build
```

FastAPI:

```text
http://localhost:8000
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

# 💻 Running Without Docker

Install dependencies:

```bash
pip install -r requirements.txt
```

Start FastAPI:

```bash
uvicorn app.main:app --reload
```

Start Dashboard:

```bash
streamlit run dashboard/app.py
```

---

# 🎥 Running The Detection Pipeline

Process CCTV footage:

```bash
python pipeline/track_people.py
```

The pipeline:

- Detects people using YOLOv8
- Tracks visitors across frames
- Generates store events
- Stores results in SQLite

Generated events are persisted in:

```text
store.db
```

---

# 🔌 API Endpoints

## Event APIs

- GET /events
- POST /events
- POST /events/ingest

## Analytics APIs

- GET /metrics
- GET /crowding
- GET /anomaly
- GET /funnel
- GET /kpis
- GET /queue
- GET /zones
- GET /pos

### Queue Analytics

`GET /queue`

Example Response:

```json
{
  "completed": 5,
  "abandoned": 2
}
```

### Zone Analytics

`GET /zones`

Example Response:

```json
{
  "zone_entries": 12,
  "zone_exits": 10
}
```

### POS Analytics

`GET /pos`

Example Response:

```json
{
  "transactions": 3,
  "revenue": 2700,
  "avg_basket_value": 900
}
```

## Health API

- GET /health

## Store Analytics

```text
GET /stores/{store_id}/metrics
GET /stores/{store_id}/funnel
```

Example:

```text
GET /stores/STORE_BLR_002/metrics
GET /stores/STORE_BLR_002/funnel
```
---

# ❤️ Health Monitoring

Health endpoint verifies:

- API Availability
- Database Connectivity
- Service Status

Example Response:

```json
{
  "status": "healthy",
  "database": "connected",
  "api": "running",
  "service": "store-intelligence-system"
}
```

---

# 📄 Documentation

Additional documentation included:

- DESIGN.md
- CHOICES.md

These documents explain:

- Architecture Decisions
- Engineering Tradeoffs
- Scalability Considerations
- Future Improvements

---

# 🔮 Future Improvements

- Multi-Camera Tracking & Re-Identification
- PostgreSQL Integration
- Real-Time Event Streaming
- Visitor-to-POS Attribution
- Staff/Customer Differentiation
- Age & Gender Prediction
- Advanced Visitor Re-Identification
- Cloud Deployment

---

# 🛠️ Tech Stack

- Python
- YOLOv8
- OpenCV
- FastAPI
- Streamlit
- SQLite
- SQLAlchemy
- Docker
- Swagger/OpenAPI
---

# 👩‍💻 Author

**Pavani Ravuvari**

Purplle Store Intelligence Challenge 2026
