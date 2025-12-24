#!/usr/bin/env python3
"""
Create sample data for testing
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.database import SessionLocal
from app.services.dataset_service import DatasetService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_datasets():
    """Create sample datasets"""
    db = SessionLocal()

    try:
        # Create a sample dataset
        logger.info("Creating sample dataset...")
        dataset = DatasetService.create_dataset(
            db=db,
            name="sample_qa_dataset",
            dataset_type="alpaca",
            description="Sample Q&A dataset for testing",
        )
        logger.info(f"✅ Created dataset: {dataset['name']} (ID: {dataset['id']})")

        # Add sample data
        samples = [
            {
                "instruction": "What is machine learning?",
                "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            },
            {
                "instruction": "Explain deep learning in simple terms",
                "output": "Deep learning is a type of machine learning that uses neural networks with multiple layers to learn complex patterns in data, similar to how the human brain processes information.",
            },
            {
                "instruction": "What is the difference between supervised and unsupervised learning?",
                "output": "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data without predefined categories.",
            },
            {
                "instruction": "What is a neural network?",
                "output": "A neural network is a computational model inspired by biological neurons that processes information through interconnected layers of nodes to learn patterns and make predictions.",
            },
            {
                "instruction": "Explain the concept of overfitting",
                "output": "Overfitting occurs when a machine learning model learns the training data too well, including noise and outliers, resulting in poor performance on new, unseen data.",
            },
        ]

        logger.info("Adding sample data...")
        for sample in samples:
            DatasetService.add_sample(
                db=db,
                dataset_id=dataset["id"],
                instruction=sample["instruction"],
                output=sample["output"],
            )

        logger.info(f"✅ Added {len(samples)} samples to dataset")

        # Create another dataset
        logger.info("Creating second sample dataset...")
        dataset2 = DatasetService.create_dataset(
            db=db,
            name="coding_questions",
            dataset_type="alpaca",
            description="Sample coding questions dataset",
        )

        coding_samples = [
            {
                "instruction": "Write a Python function to reverse a string",
                "output": "def reverse_string(s):\n    return s[::-1]",
            },
            {
                "instruction": "How do you create a list in Python?",
                "output": "You can create a list using square brackets: my_list = [1, 2, 3, 4, 5]",
            },
            {
                "instruction": "Explain what a dictionary is in Python",
                "output": "A dictionary is a collection of key-value pairs enclosed in curly braces, like {'name': 'John', 'age': 30}. It allows fast lookup by key.",
            },
        ]

        for sample in coding_samples:
            DatasetService.add_sample(
                db=db,
                dataset_id=dataset2["id"],
                instruction=sample["instruction"],
                output=sample["output"],
            )

        logger.info(f"✅ Added {len(coding_samples)} samples to second dataset")
        logger.info("🎉 Sample data created successfully!")

    except Exception as e:
        logger.error(f"❌ Error creating sample data: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    create_sample_datasets()
