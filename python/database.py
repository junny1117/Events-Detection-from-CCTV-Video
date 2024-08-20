from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

engine = create_engine('sqlite:///site.db')
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()
Base.query = db_session.query_property()

class Event(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    label = Column(String(50))
    confidence = Column(Float)
    timestamp = Column(DateTime)

class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(512))

def init_db():
    Base.metadata.create_all(bind=engine)
