from sqlalchemy import Column, Integer, String, Float, DateTime
from python.database import Base

class Event(Base):
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True)
    label = Column(String(50))
    confidence = Column(Float)
    timestamp = Column(DateTime)
    
    def __init__(self, label, confidence, timestamp):
        self.label = label
        self.confidence = confidence
        self.timestamp = timestamp

    def __repr__(self):
        return f"<Event(id={self.id}, label='{self.label}', confidence={self.confidence}, timestamp={self.timestamp})>"
