#!/usr/bin/env python3
"""
LLaMA Factory Data Management Application
Main entry point for the Gradio UI
"""

import gradio as gr
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.database import init_db
from app.ui.dataset_ui import create_dataset_ui
from app.ui.training_ui import create_training_ui
from app.ui.model_ui import create_model_ui
from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_app():
    """Create and configure the Gradio application"""

    # Initialize database
    logger.info("Initializing database...")
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    # Create Gradio interface
    with gr.Blocks(
        title="LLaMA Factory Data Management",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        """,
    ) as app:
        gr.Markdown(
            """
        # 🦙 LLaMA Factory Data Management System

        Comprehensive tool for managing datasets, training models, and deploying fine-tuned LLMs
        """
        )

        # Create tabs
        create_dataset_ui()
        create_training_ui()
        create_model_ui()

        # Add footer
        gr.Markdown(
            """
        ---
        **LLaMA Factory Data Management v0.1.0** | Built with Gradio + PostgreSQL
        """
        )

    return app


def main():
    """Main entry point"""
    logger.info("Starting LLaMA Factory Data Management System...")
    logger.info(f"Database URL: {settings.database_url}")
    logger.info(f"Server: http://{settings.APP_HOST}:{settings.APP_PORT}")

    app = create_app()

    # Launch the app
    app.launch(
        server_name=settings.APP_HOST,
        server_port=settings.APP_PORT,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
