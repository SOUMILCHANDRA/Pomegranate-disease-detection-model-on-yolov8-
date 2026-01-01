from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import datetime
import os

# --- Configuration ---
# Switched to SQLite for zero-setup persistence.
# This creates a file 'pomegranate.db' in the same folder.
DB_CONNECTION_STRING = "sqlite:///pomegranate.db"

Base = declarative_base()

# --- database Models ---

class ScanResult(Base):
    __tablename__ = 'scan_results'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    total_plants = Column(Integer)
    status = Column(String(50)) # Healthy, Infected
    
    # Relationships
    diseases = relationship("DiseaseDetection", back_populates="scan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "total_plants": self.total_plants,
            "status": self.status,
            "diseases": [d.to_dict() for d in self.diseases]
        }

class DiseaseDetection(Base):
    __tablename__ = 'disease_detections'
    
    id = Column(Integer, primary_key=True)
    scan_id = Column(Integer, ForeignKey('scan_results.id'))
    disease_name = Column(String(100))
    avg_infection = Column(Float)
    
    # Relationships
    scan = relationship("ScanResult", back_populates="diseases")
    prescriptions = relationship("Prescription", back_populates="disease")

    def to_dict(self):
        return {
            "disease_name": self.disease_name,
            "avg_infection": self.avg_infection,
            "prescriptions": [p.to_dict() for p in self.prescriptions]
        }

class Prescription(Base):
    __tablename__ = 'prescriptions'
    
    id = Column(Integer, primary_key=True)
    disease_id = Column(Integer, ForeignKey('disease_detections.id'))
    company = Column(String(100))
    chemical = Column(String(200))
    dosage = Column(Float)
    unit = Column(String(50))
    price_rs = Column(String(50))
    
    # Relationships
    disease = relationship("DiseaseDetection", back_populates="prescriptions")

    def to_dict(self):
        return {
            "company": self.company,
            "chemical": self.chemical,
            "dosage": self.dosage,
            "unit": self.unit,
            "price_rs": self.price_rs
        }

# --- Connector Class ---

class PostgresConnector:
    def __init__(self, connection_string=DB_CONNECTION_STRING):
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create Tables if they don't exist
        try:
            Base.metadata.create_all(self.engine)
            print("[Postgres] Tables created/verified.")
        except Exception as e:
            print(f"[Postgres] Error connecting/creating tables: {e}")

    def save_report(self, report_json):
        """
        Parses the specific JSON structure from main_analyzer and saves to DB.
        """
        session = self.Session()
        try:
            # 1. Create Scan Result
            scan = ScanResult(
                total_plants=report_json.get("total_plants", 0),
                status=report_json.get("status", "Unknown")
            )
            session.add(scan)
            session.flush() # Get ID
            
            # 2. Add Diseases
            for d_data in report_json.get("detected_diseases", []):
                disease = DiseaseDetection(
                    scan_id=scan.id,
                    disease_name=d_data["disease_name"],
                    avg_infection=d_data["avg_infection"]
                )
                session.add(disease)
                session.flush() # Get ID
                
                # 3. Add Prescriptions
                for p_data in d_data.get("prescriptions", []):
                    rx = Prescription(
                        disease_id=disease.id,
                        company=p_data["company"],
                        chemical=p_data["chemical"],
                        dosage=p_data["dosage"],
                        unit=p_data["unit"],
                        price_rs=str(p_data["price_rs"])
                    )
                    session.add(rx)
            
            session.commit()
            print(f"[Postgres] Report Saved Successfully! ID: {scan.id}")
            return scan.id
        except Exception as e:
            session.rollback()
            print(f"[Postgres] Save Failed: {e}")
            return None
        finally:
            session.close()

    def get_latest_reports(self, limit=10):
        session = self.Session()
        try:
            scans = session.query(ScanResult).order_by(ScanResult.timestamp.desc()).limit(limit).all()
            return [s.to_dict() for s in scans]
        finally:
            session.close()
