import os
import json
import subprocess
import threading
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from app.database import crud
from app.database.models import TrainingStatus
from config import settings


class TrainingService:
    """Service for managing training jobs"""

    active_trainings: Dict[int, subprocess.Popen] = {}

    @staticmethod
    def create_training_job(
        db: Session,
        name: str,
        dataset_id: int,
        base_model: str,
        description: Optional[str] = None,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 8,
        max_seq_length: int = 512,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> dict:
        """Create a new training job"""
        job = crud.create_training_job(
            db=db,
            name=name,
            dataset_id=dataset_id,
            base_model=base_model,
            description=description,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_seq_length=max_seq_length,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        return {
            "id": job.id,
            "name": job.name,
            "status": job.status.value,
            "dataset_id": job.dataset_id,
            "base_model": job.base_model,
        }

    @staticmethod
    def get_training_jobs(db: Session, status: Optional[str] = None) -> List[dict]:
        """Get all training jobs"""
        status_enum = None
        if status:
            try:
                status_enum = TrainingStatus[status.upper()]
            except KeyError:
                pass

        jobs = crud.get_training_jobs(db, status=status_enum)
        return [
            {
                "id": job.id,
                "name": job.name,
                "status": job.status.value,
                "dataset_id": job.dataset_id,
                "base_model": job.base_model,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat()
                if job.completed_at
                else None,
            }
            for job in jobs
        ]

    @staticmethod
    def start_training(db: Session, job_id: int) -> dict:
        """Start a training job"""
        job = crud.get_training_job(db, job_id)
        if not job:
            raise ValueError(f"Training job {job_id} not found")

        if job.status != TrainingStatus.PENDING:
            raise ValueError(
                f"Training job {job_id} is not in PENDING status (current: {job.status.value})"
            )

        # Check concurrent training limit
        active_count = len(
            [j for j in TrainingService.active_trainings.values() if j.poll() is None]
        )
        if active_count >= settings.MAX_CONCURRENT_TRAININGS:
            raise ValueError(
                f"Maximum concurrent trainings ({settings.MAX_CONCURRENT_TRAININGS}) reached"
            )

        # Update status to RUNNING
        crud.update_training_job_status(db, job_id, TrainingStatus.RUNNING)

        # Prepare output directory
        output_dir = os.path.join(settings.TRAINED_MODEL_PATH, job.name)
        os.makedirs(output_dir, exist_ok=True)

        # Export dataset for training
        dataset = crud.get_dataset(db, job.dataset_id)
        dataset_file = os.path.join(settings.DATASET_PATH, f"{dataset.name}.json")

        # Export samples to JSON
        from .dataset_service import DatasetService

        DatasetService.export_to_json(db, job.dataset_id, dataset_file)

        # Prepare training config
        training_config = {
            "model_name_or_path": job.base_model,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "lora_rank": job.lora_rank,
            "lora_alpha": job.lora_alpha,
            "lora_dropout": job.lora_dropout,
            "dataset": dataset.name,
            "dataset_dir": settings.DATASET_PATH,
            "template": "default",
            "cutoff_len": job.max_seq_length,
            "learning_rate": job.learning_rate,
            "num_train_epochs": job.num_epochs,
            "per_device_train_batch_size": job.batch_size,
            "gradient_accumulation_steps": job.gradient_accumulation_steps,
            "save_steps": 500,
            "output_dir": output_dir,
            "logging_steps": 10,
            "plot_loss": True,
        }

        # Start training in a separate thread
        def run_training():
            try:
                # This is a simplified example - you would call LLaMA Factory's training script here
                # For now, we'll simulate a training process
                import time

                time.sleep(5)  # Simulate training

                # Update job status
                crud.update_training_job_status(db, job_id, TrainingStatus.COMPLETED)
                crud.update_training_job_results(
                    db, job_id, output_dir, best_metric=0.95, final_loss=0.1
                )

            except Exception as e:
                crud.update_training_job_status(
                    db, job_id, TrainingStatus.FAILED, error_message=str(e)
                )

        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()

        return {
            "job_id": job_id,
            "status": "started",
            "message": "Training job started successfully",
        }

    @staticmethod
    def stop_training(db: Session, job_id: int) -> dict:
        """Stop a running training job"""
        job = crud.get_training_job(db, job_id)
        if not job:
            raise ValueError(f"Training job {job_id} not found")

        if job.status != TrainingStatus.RUNNING:
            raise ValueError(f"Training job {job_id} is not running")

        # Stop the process if it exists
        if job_id in TrainingService.active_trainings:
            process = TrainingService.active_trainings[job_id]
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)

        # Update status
        crud.update_training_job_status(db, job_id, TrainingStatus.CANCELLED)

        return {"job_id": job_id, "status": "cancelled"}

    @staticmethod
    def get_training_logs(db: Session, job_id: int) -> List[str]:
        """Get training logs"""
        job = crud.get_training_job(db, job_id)
        if not job:
            raise ValueError(f"Training job {job_id} not found")

        log_file = os.path.join(job.output_dir or "", "training.log")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                return f.readlines()

        return ["No logs available yet"]

    @staticmethod
    def delete_training_job(db: Session, job_id: int) -> bool:
        """Delete a training job"""
        return crud.delete_training_job(db, job_id)
