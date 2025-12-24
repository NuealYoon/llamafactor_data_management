import json
import os
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
import pandas as pd

from app.database import crud
from app.database.models import DatasetType
from config import settings


class DatasetService:
    """Service for managing datasets"""

    @staticmethod
    def create_dataset(
        db: Session,
        name: str,
        dataset_type: str,
        description: Optional[str] = None,
    ) -> dict:
        """Create a new dataset"""
        try:
            dataset_type_enum = DatasetType[dataset_type.upper()]
        except KeyError:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        dataset = crud.create_dataset(
            db=db,
            name=name,
            dataset_type=dataset_type_enum,
            description=description,
        )

        return {
            "id": dataset.id,
            "name": dataset.name,
            "type": dataset.dataset_type.value,
            "description": dataset.description,
            "total_samples": dataset.total_samples,
        }

    @staticmethod
    def get_datasets(db: Session) -> List[dict]:
        """Get all datasets"""
        datasets = crud.get_datasets(db)
        return [
            {
                "id": ds.id,
                "name": ds.name,
                "type": ds.dataset_type.value,
                "description": ds.description,
                "total_samples": ds.total_samples,
                "created_at": ds.created_at.isoformat() if ds.created_at else None,
            }
            for ds in datasets
        ]

    @staticmethod
    def add_sample(
        db: Session,
        dataset_id: int,
        instruction: str,
        output: str,
        input_text: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Add a sample to a dataset"""
        sample = crud.create_dataset_sample(
            db=db,
            dataset_id=dataset_id,
            instruction=instruction,
            output_text=output,
            input_text=input_text,
            system_prompt=system_prompt,
        )

        return {
            "id": sample.id,
            "dataset_id": sample.dataset_id,
            "instruction": sample.instruction,
            "input": sample.input_text,
            "output": sample.output_text,
            "system": sample.system_prompt,
        }

    @staticmethod
    def get_samples(
        db: Session, dataset_id: int, skip: int = 0, limit: int = 100
    ) -> List[dict]:
        """Get samples from a dataset"""
        samples = crud.get_dataset_samples(db, dataset_id, skip, limit)
        return [
            {
                "id": s.id,
                "instruction": s.instruction,
                "input": s.input_text,
                "output": s.output_text,
                "system": s.system_prompt,
            }
            for s in samples
        ]

    @staticmethod
    def import_from_json(
        db: Session, dataset_id: int, file_path: str
    ) -> Dict[str, int]:
        """Import samples from JSON file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of samples")

        imported_count = 0
        for item in data:
            try:
                crud.create_dataset_sample(
                    db=db,
                    dataset_id=dataset_id,
                    instruction=item.get("instruction", ""),
                    output_text=item.get("output", ""),
                    input_text=item.get("input"),
                    system_prompt=item.get("system"),
                )
                imported_count += 1
            except Exception as e:
                print(f"Error importing sample: {e}")
                continue

        return {"imported": imported_count, "total": len(data)}

    @staticmethod
    def export_to_json(db: Session, dataset_id: int, output_path: str) -> str:
        """Export dataset to JSON file"""
        samples = crud.get_dataset_samples(db, dataset_id, skip=0, limit=10000)

        data = []
        for sample in samples:
            item = {
                "instruction": sample.instruction,
                "output": sample.output_text,
            }
            if sample.input_text:
                item["input"] = sample.input_text
            if sample.system_prompt:
                item["system"] = sample.system_prompt

            data.append(item)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return output_path

    @staticmethod
    def delete_dataset(db: Session, dataset_id: int) -> bool:
        """Delete a dataset"""
        return crud.delete_dataset(db, dataset_id)

    @staticmethod
    def delete_sample(db: Session, sample_id: int) -> bool:
        """Delete a sample"""
        return crud.delete_dataset_sample(db, sample_id)

    @staticmethod
    def get_dataset_stats(db: Session, dataset_id: int) -> dict:
        """Get dataset statistics"""
        dataset = crud.get_dataset(db, dataset_id)
        if not dataset:
            return {}

        samples = crud.get_dataset_samples(db, dataset_id, skip=0, limit=10000)

        total_chars = sum(
            len(s.instruction) + len(s.output_text or "") + len(s.input_text or "")
            for s in samples
        )

        return {
            "total_samples": dataset.total_samples,
            "total_characters": total_chars,
            "avg_chars_per_sample": total_chars / max(dataset.total_samples, 1),
        }
