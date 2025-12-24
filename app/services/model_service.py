import os
from typing import List, Optional
from sqlalchemy.orm import Session

from app.database import crud
from config import settings


class ModelService:
    """Service for managing trained models"""

    @staticmethod
    def create_model_from_training(
        db: Session,
        training_job_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> dict:
        """Create a trained model from a training job"""
        job = crud.get_training_job(db, training_job_id)
        if not job:
            raise ValueError(f"Training job {training_job_id} not found")

        if not job.output_dir:
            raise ValueError("Training job has no output directory")

        model_name = name or f"{job.name}_model"

        model = crud.create_trained_model(
            db=db,
            name=model_name,
            training_job_id=training_job_id,
            model_path=job.output_dir,
            base_model=job.base_model,
            description=description,
            eval_loss=job.final_loss,
        )

        return {
            "id": model.id,
            "name": model.name,
            "training_job_id": model.training_job_id,
            "model_path": model.model_path,
            "base_model": model.base_model,
        }

    @staticmethod
    def get_models(db: Session, is_active: Optional[bool] = None) -> List[dict]:
        """Get all trained models"""
        models = crud.get_trained_models(db, is_active=is_active)
        return [
            {
                "id": model.id,
                "name": model.name,
                "base_model": model.base_model,
                "model_path": model.model_path,
                "is_active": model.is_active,
                "eval_loss": model.eval_loss,
                "created_at": model.created_at.isoformat()
                if model.created_at
                else None,
            }
            for model in models
        ]

    @staticmethod
    def get_model(db: Session, model_id: int) -> Optional[dict]:
        """Get a trained model by ID"""
        model = crud.get_trained_model(db, model_id)
        if not model:
            return None

        return {
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "base_model": model.base_model,
            "model_path": model.model_path,
            "training_job_id": model.training_job_id,
            "is_active": model.is_active,
            "eval_loss": model.eval_loss,
            "eval_accuracy": model.eval_accuracy,
            "created_at": model.created_at.isoformat() if model.created_at else None,
        }

    @staticmethod
    def update_model(
        db: Session,
        model_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[dict]:
        """Update a trained model"""
        model = crud.update_trained_model(
            db=db,
            model_id=model_id,
            name=name,
            description=description,
            is_active=is_active,
        )

        if not model:
            return None

        return {
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "is_active": model.is_active,
        }

    @staticmethod
    def delete_model(db: Session, model_id: int, delete_files: bool = False) -> bool:
        """Delete a trained model"""
        if delete_files:
            model = crud.get_trained_model(db, model_id)
            if model and model.model_path and os.path.exists(model.model_path):
                import shutil

                shutil.rmtree(model.model_path)

        return crud.delete_trained_model(db, model_id)

    @staticmethod
    def load_model_for_inference(db: Session, model_id: int):
        """Load a model for inference (placeholder for future implementation)"""
        model = crud.get_trained_model(db, model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")

        # This would load the model using transformers/LLaMA Factory
        # For now, just return model info
        return {
            "model_id": model.id,
            "model_path": model.model_path,
            "status": "loaded",
        }
