from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import declarative_base, sessionmaker

engine = create_engine("sqlite:///data/exoplanets.db", echo=False, future=True)
Base = declarative_base()

class Planet(Base):
    __tablename__ = "planet"
    id = Column(Integer, primary_key=True)
    name = Column(String, index=True)
    pl_rade = Column(Float, nullable=False)
    pl_bmasse = Column(Float, nullable=False)
    pl_orbsmax = Column(Float, nullable=False)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def init_db():
    Base.metadata.create_all(engine)
