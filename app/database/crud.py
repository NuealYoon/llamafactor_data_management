from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import json

from .models import (
    Dataset,
    DatasetSample,
    TrainingJob,
    TrainedModel,
    DatasetType,
    TrainingStatus,
)


# ==================== Dataset CRUD ====================

def create_dataset(
    db: Session,
    name: str,
    dataset_type: DatasetType,
    description: Optional[str] = None,
    file_path: Optional[str] = None,
) -> Dataset:
    """Create a new dataset"""
    dataset = Dataset(
        name=name,
        description=description,
        dataset_type=dataset_type,
        file_path=file_path,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def get_dataset(db: Session, dataset_id: int) -> Optional[Dataset]:
    """Get dataset by ID"""
    return db.query(Dataset).filter(Dataset.id == dataset_id).first()


def get_dataset_by_name(db: Session, name: str) -> Optional[Dataset]:
    """Get dataset by name"""
    return db.query(Dataset).filter(Dataset.name == name).first()


def get_datasets(db: Session, skip: int = 0, limit: int = 100) -> List[Dataset]:
    """Get all datasets"""
    return db.query(Dataset).offset(skip).limit(limit).all()


def update_dataset(
    db: Session,
    dataset_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[Dataset]:
    """Update dataset"""
    dataset = get_dataset(db, dataset_id)
    if dataset:
        if name:
            dataset.name = name
        if description is not None:
            dataset.description = description
        dataset.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(dataset)
    return dataset


def delete_dataset(db: Session, dataset_id: int) -> bool:
    """Delete dataset"""
    dataset = get_dataset(db, dataset_id)
    if dataset:
        db.delete(dataset)
        db.commit()
        return True
    return False


# ==================== Dataset Sample CRUD ====================

def create_dataset_sample(
    db: Session,
    dataset_id: int,
    instruction: str,
    output_text: str,
    input_text: Optional[str] = None,
    system_prompt: Optional[str] = None,
    additional_data: Optional[dict] = None,
) -> DatasetSample:
    """Create a new dataset sample"""
    sample = DatasetSample(
        dataset_id=dataset_id,
        instruction=instruction,
        input_text=input_text,
        output_text=output_text,
        system_prompt=system_prompt,
        additional_data=additional_data,
    )
    db.add(sample)

    # Update total samples count
    dataset = get_dataset(db, dataset_id)
    if dataset:
        dataset.total_samples += 1

    db.commit()
    db.refresh(sample)
    return sample


def get_dataset_samples(
    db: Session, dataset_id: int, skip: int = 0, limit: int = 100
) -> List[DatasetSample]:
    """Get samples for a dataset"""
    return (
        db.query(DatasetSample)
        .filter(DatasetSample.dataset_id == dataset_id)
        .offset(skip)
        .limit(limit)
        .all()
    )


def delete_dataset_sample(db: Session, sample_id: int) -> bool:
    """Delete a dataset sample"""
    sample = db.query(DatasetSample).filter(DatasetSample.id == sample_id).first()
    if sample:
        dataset_id = sample.dataset_id
        db.delete(sample)

        # Update total samples count
        dataset = get_dataset(db, dataset_id)
        if dataset:
            dataset.total_samples = max(0, dataset.total_samples - 1)

        db.commit()
        return True
    return False


# ==================== Training Job CRUD ====================

def create_training_job(
    db: Session,
    name: str,
    dataset_id: int,
    base_model: str,
    description: Optional[str] = None,
    **training_params,
) -> TrainingJob:
    """Create a new training job"""
    job = TrainingJob(
        name=name,
        description=description,
        dataset_id=dataset_id,
        base_model=base_model,
        **training_params,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_training_job(db: Session, job_id: int) -> Optional[TrainingJob]:
    """Get training job by ID"""
    return db.query(TrainingJob).filter(TrainingJob.id == job_id).first()


def get_training_jobs(
    db: Session, skip: int = 0, limit: int = 100, status: Optional[TrainingStatus] = None
) -> List[TrainingJob]:
    """Get all training jobs"""
    query = db.query(TrainingJob)
    if status:
        query = query.filter(TrainingJob.status == status)
    return query.offset(skip).limit(limit).all()


def update_training_job_status(
    db: Session,
    job_id: int,
    status: TrainingStatus,
    error_message: Optional[str] = None,
) -> Optional[TrainingJob]:
    """Update training job status"""
    job = get_training_job(db, job_id)
    if job:
        job.status = status
        if status == TrainingStatus.RUNNING and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
            job.completed_at = datetime.utcnow()
        if error_message:
            job.error_message = error_message
        db.commit()
        db.refresh(job)
    return job


def update_training_job_results(
    db: Session,
    job_id: int,
    output_dir: str,
    best_metric: Optional[float] = None,
    final_loss: Optional[float] = None,
) -> Optional[TrainingJob]:
    """Update training job results"""
    job = get_training_job(db, job_id)
    if job:
        job.output_dir = output_dir
        if best_metric is not None:
            job.best_metric = best_metric
        if final_loss is not None:
            job.final_loss = final_loss
        db.commit()
        db.refresh(job)
    return job


def delete_training_job(db: Session, job_id: int) -> bool:
    """Delete training job"""
    job = get_training_job(db, job_id)
    if job:
        db.delete(job)
        db.commit()
        return True
    return False


# ==================== Trained Model CRUD ====================

def create_trained_model(
    db: Session,
    name: str,
    training_job_id: int,
    model_path: str,
    base_model: str,
    description: Optional[str] = None,
    **model_params,
) -> TrainedModel:
    """Create a new trained model"""
    model = TrainedModel(
        name=name,
        description=description,
        training_job_id=training_job_id,
        model_path=model_path,
        base_model=base_model,
        **model_params,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def get_trained_model(db: Session, model_id: int) -> Optional[TrainedModel]:
    """Get trained model by ID"""
    return db.query(TrainedModel).filter(TrainedModel.id == model_id).first()


def get_trained_models(
    db: Session, skip: int = 0, limit: int = 100, is_active: Optional[bool] = None
) -> List[TrainedModel]:
    """Get all trained models"""
    query = db.query(TrainedModel)
    if is_active is not None:
        query = query.filter(TrainedModel.is_active == is_active)
    return query.offset(skip).limit(limit).all()


def update_trained_model(
    db: Session,
    model_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> Optional[TrainedModel]:
    """Update trained model"""
    model = get_trained_model(db, model_id)
    if model:
        if name:
            model.name = name
        if description is not None:
            model.description = description
        if is_active is not None:
            model.is_active = is_active
        model.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(model)
    return model


def delete_trained_model(db: Session, model_id: int) -> bool:
    """Delete trained model"""
    model = get_trained_model(db, model_id)
    if model:
        db.delete(model)
        db.commit()
        return True
    return False
