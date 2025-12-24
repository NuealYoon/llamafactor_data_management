from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum, Float, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


class DatasetType(enum.Enum):
    """Dataset type enumeration"""
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    CUSTOM = "custom"


class TrainingStatus(enum.Enum):
    """Training job status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Dataset(Base):
    """Dataset model"""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    dataset_type = Column(Enum(DatasetType), nullable=False, default=DatasetType.ALPACA)
    file_path = Column(String(512), nullable=True)
    total_samples = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    samples = relationship("DatasetSample", back_populates="dataset", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="dataset")

    def __repr__(self):
        return f"<Dataset(id={self.id}, name='{self.name}', type={self.dataset_type})>"


class DatasetSample(Base):
    """Dataset sample model"""
    __tablename__ = "dataset_samples"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    instruction = Column(Text, nullable=False)
    input_text = Column(Text, nullable=True)
    output_text = Column(Text, nullable=False)
    system_prompt = Column(Text, nullable=True)
    additional_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    dataset = relationship("Dataset", back_populates="samples")

    def __repr__(self):
        return f"<DatasetSample(id={self.id}, dataset_id={self.dataset_id})>"


class TrainingJob(Base):
    """Training job model"""
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    base_model = Column(String(512), nullable=False)
    status = Column(Enum(TrainingStatus), nullable=False, default=TrainingStatus.PENDING)

    # Training parameters
    learning_rate = Column(Float, default=5e-5)
    num_epochs = Column(Integer, default=3)
    batch_size = Column(Integer, default=4)
    gradient_accumulation_steps = Column(Integer, default=8)
    max_seq_length = Column(Integer, default=512)
    lora_rank = Column(Integer, default=8)
    lora_alpha = Column(Integer, default=16)
    lora_dropout = Column(Float, default=0.05)

    # Training configuration
    training_config = Column(JSON, nullable=True)

    # Results
    output_dir = Column(String(512), nullable=True)
    best_metric = Column(Float, nullable=True)
    final_loss = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    dataset = relationship("Dataset", back_populates="training_jobs")
    model = relationship("TrainedModel", back_populates="training_job", uselist=False)

    def __repr__(self):
        return f"<TrainingJob(id={self.id}, name='{self.name}', status={self.status})>"


class TrainedModel(Base):
    """Trained model model"""
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=False, unique=True, index=True)
    model_path = Column(String(512), nullable=False)
    base_model = Column(String(512), nullable=False)

    # Model metadata
    model_size = Column(String(50), nullable=True)  # e.g., "7B", "13B"
    quantization = Column(String(50), nullable=True)  # e.g., "4bit", "8bit"

    # Performance metrics
    eval_loss = Column(Float, nullable=True)
    eval_accuracy = Column(Float, nullable=True)

    # Model configuration
    model_config = Column(JSON, nullable=True)

    # Status
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    training_job = relationship("TrainingJob", back_populates="model")

    def __repr__(self):
        return f"<TrainedModel(id={self.id}, name='{self.name}', path='{self.model_path}')>"
