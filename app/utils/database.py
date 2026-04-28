import sqlite3
import os
from flask import current_app
import logging
logger = logging.getLogger(__name__)

def init_db():
    database_dir = current_app.config['DATABASE_FOLDER']
    
    os.makedirs(database_dir, exist_ok=True)
    db_path = os.path.join(database_dir, 'finance_app.db')
    
    logger.debug(f"db_path: {db_path}")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Create the transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cash_flow (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_date TEXT NOT NULL,
                item_description TEXT NOT NULL,
                amount_eur REAL NOT NULL,
                frequency TEXT NOT NULL,
                category TEXT NOT NULL,
                source_type TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT
            )
        ''')
        conn.commit()

if __name__ == "__main__":
    init_db()
    logger.info("Database initialized.")
    
