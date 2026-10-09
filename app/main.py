from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from app.database import engine, get_db
from app.models import Base, Event, EventCreate
from app.pos import get_pos_metrics

Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Purplle Store Intelligence API"}

@app.post("/events")
def create_event(event: EventCreate, db: Session = Depends(get_db)):

    new_event = Event(
        event_type=event.event_type,
        person_id=event.person_id,
        zone=event.zone,
        timestamp=event.timestamp
    )

    db.add(new_event)
    db.commit()
    db.refresh(new_event)

    return {
        "status": "saved",
        "id": new_event.id
    }
@app.get("/events")
def get_events(db: Session = Depends(get_db)):
    events = db.query(Event).all()

    return [
        {
            "id": e.id,
            "event_type": e.event_type,
            "person_id": e.person_id,
            "zone": e.zone,
            "timestamp": e.timestamp
        }
        for e in events
    ]
@app.get("/metrics")
def metrics(db: Session = Depends(get_db)):

    total_events = db.query(Event).count()

    entries = db.query(Event)\
        .filter(Event.event_type == "entry")\
        .count()

    exits = db.query(Event)\
        .filter(Event.event_type == "exit")\
        .count()

    return {
        "total_events": total_events,
        "entries": entries,
        "exits": exits
    }
from app.analytics import get_total_customers
from app.crowding import check_crowding

@app.get("/crowding")
def crowding():
    return check_crowding()
from app.anomaly import detect_anomaly

@app.get("/anomaly")
def anomaly():
    return detect_anomaly()
@app.get("/funnel")
def funnel():

    from app.database import SessionLocal
    from app.models import Event

    db = SessionLocal()

    entries = db.query(Event).filter(
        Event.event_type == "entry"
    ).count()

    exits = db.query(Event).filter(
        Event.event_type == "exit"
    ).count()

    active = entries - exits

    db.close()

    return {
        "entered": entries,
        "active": active,
        "exited": exits
    }
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "database": "connected",
        "api": "running",
        "service": "store-intelligence-system"
    }
from app.kpi import generate_kpis

@app.get("/kpis")
def kpis():
    return generate_kpis()
@app.post("/events/ingest")
def ingest_event(event: EventCreate,
                 db: Session = Depends(get_db)):

    new_event = Event(
        event_type=event.event_type,
        person_id=event.person_id,
        zone=event.zone,
        timestamp=event.timestamp
    )

    db.add(new_event)
    db.commit()
    db.refresh(new_event)

    return {
        "status": "ingested",
        "id": new_event.id
    }
@app.get("/stores/{store_id}/metrics")
def store_metrics(
        store_id: str,
        db: Session = Depends(get_db)):

    total_events = db.query(Event).count()

    entries = db.query(Event)\
        .filter(Event.event_type == "entry")\
        .count()

    exits = db.query(Event)\
        .filter(Event.event_type == "exit")\
        .count()

    return {
        "store_id": store_id,
        "total_events": total_events,
        "entries": entries,
        "exits": exits,
        "active_visitors": entries - exits
    }
@app.get("/stores/{store_id}/funnel")
def store_funnel(
        store_id: str,
        db: Session = Depends(get_db)):

    entries = db.query(Event)\
        .filter(Event.event_type == "entry")\
        .count()

    exits = db.query(Event)\
        .filter(Event.event_type == "exit")\
        .count()

    active = entries - exits

    return {
        "store_id": store_id,
        "entered": entries,
        "active": active,
        "exited": exits
    }
@app.get("/queue")
def queue_metrics(db: Session = Depends(get_db)):
    completed = db.query(Event).filter(
        Event.event_type == "queue_completed"
    ).count()

    abandoned = db.query(Event).filter(
        Event.event_type == "queue_abandoned"
    ).count()

    return {
        "completed": completed,
        "abandoned": abandoned
    }
@app.get("/zones")
def zone_metrics(db: Session = Depends(get_db)):

    entered = db.query(Event).filter(
        Event.event_type == "zone_entered"
    ).count()

    exited = db.query(Event).filter(
        Event.event_type == "zone_exited"
    ).count()

    return {
        "zone_entries": entered,
        "zone_exits": exited
    }
@app.get("/pos")
def pos():
    return get_pos_metrics()
@app.get("/zones")
def zone_metrics(db: Session = Depends(get_db)):

    entered = db.query(Event).filter(
        Event.event_type == "zone_entered"
    ).count()

    exited = db.query(Event).filter(
        Event.event_type == "zone_exited"
    ).count()

    return {
        "zone_entries": entered,
        "zone_exits": exited
    }