from sqlalchemy import Column, Integer, String
from pydantic import BaseModel
from app.database import Base

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String)
    person_id = Column(String)
    zone = Column(String)
    timestamp = Column(String)

class EventCreate(BaseModel):
    event_type: str
    person_id: str
    zone: str
    timestamp: str