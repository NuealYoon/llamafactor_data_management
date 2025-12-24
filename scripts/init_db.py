#!/usr/bin/env python3
"""
Database initialization script
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.database import init_db
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Initialize database"""
    logger.info("Initializing database...")
    logger.info(f"Database URL: {settings.database_url}")

    try:
        init_db()
        logger.info("✅ Database initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Error initializing database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
