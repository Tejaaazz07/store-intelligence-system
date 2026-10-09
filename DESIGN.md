# Store Intelligence System - Design Document

## Overview

This project is a lightweight store intelligence platform designed to process retail camera footage and generate business insights from visitor activity. The system combines computer vision, event processing, analytics APIs, and a dashboard to provide operational visibility for a retail store.

The primary objective is to detect people in video streams, track movement, generate entry and exit events, calculate occupancy metrics, identify crowding conditions, and expose analytics through REST APIs and dashboards.

## Architecture

The system is divided into four major components:

1. Detection Pipeline
2. Event Processing Layer
3. Analytics API (FastAPI)
4. Dashboard (Streamlit)
5. Deployment Layer

The application is containerized using Docker to ensure consistent execution across environments. Docker Compose is used to orchestrate application startup and simplify deployment.

### Detection Pipeline

The detection pipeline uses YOLOv8 for person detection. Video frames are processed sequentially and person detections are extracted. Bounding boxes with low confidence are filtered to reduce false positives.

Each detected person receives a tracking ID which is used to maintain identity across frames. This enables counting unique visitors instead of counting detections.

### Zone Tracking

The store is divided into logical zones. Based on the x-coordinate of a tracked person, the system assigns a zone label.

When a person moves from one zone to another, a ZONE_ENTER event is generated. These events provide visibility into customer movement patterns and dwell behavior.

### Event Processing

All activity is represented as events. Examples include:

- ENTRY
- EXIT
- REENTRY
- ZONE_ENTER

The event-driven architecture simplifies downstream analytics because every metric can be derived from event history.

Events are stored in a SQLite database and can be queried through API endpoints.

### Analytics Layer

The FastAPI application exposes business metrics through REST endpoints.

Examples include:

- Total visitors
- Current occupancy
- Entry count
- Exit count
- Funnel metrics
- Crowd status
- Anomaly detection
- KPI summaries

This separation allows analytics to evolve independently from the detection pipeline.

### Dashboard

A Streamlit dashboard provides a visual interface for operational monitoring.

The dashboard displays:

- Total events
- Entries
- Exits
- Occupancy
- Crowding status
- Anomaly alerts

This enables non-technical users to monitor store activity without interacting with the API directly.

## Data Storage

SQLite was selected for simplicity and portability. It requires no external setup and is sufficient for the scale of this challenge.

All events are persisted and remain available for later analysis.

## Scalability Considerations

The architecture is intentionally modular.

Future improvements may include:

- PostgreSQL for production workloads
- Kafka-based event streaming
- Multi-camera synchronization
- Cross-camera visitor deduplication
- Real-time processing pipelines
- Cloud deployment

## Reliability

The application includes a health endpoint and containerized deployment using Docker. This allows consistent execution across development and evaluation environments.

## POS Transaction Correlation

The challenge resources included POS transaction data containing store-level purchase records such as transaction timestamps, basket values, and product information.

In a production-grade Store Intelligence System, these transactions can be correlated with visitor sessions to calculate business metrics such as conversion rate, billing-zone effectiveness, and purchase funnel performance.

The current implementation focuses on visitor detection, tracking, event generation, occupancy analytics, anomaly detection, and KPI generation. Direct correlation between visitor sessions and POS transactions was identified as a future enhancement.

A future version of the system can match visitor sessions with nearby transaction timestamps and billing-zone activity to estimate customer conversion and generate richer retail intelligence metrics.

## Conclusion

The design prioritizes simplicity, maintainability, and ease of deployment while satisfying the functional requirements of the challenge.

The system demonstrates an end-to-end retail intelligence workflow, including visitor detection, tracking, event generation, analytics computation, anomaly detection, KPI reporting, REST APIs, and dashboard visualization.

The architecture is modular and can be extended with advanced capabilities such as POS transaction correlation, staff identification, cross-camera re-identification, event streaming, and cloud-native deployment for production-scale retail environments.