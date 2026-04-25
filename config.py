import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Get the absolute path of the directory where config.py is located
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_FOLDER = os.path.join(BASE_DIR, 'data')
    TEST_FOLDER = os.path.join(BASE_DIR, 'test')
    DATABASE_FOLDER = os.path.join(BASE_DIR, 'instance')
    USER_CONFIG_PATH = os.path.join(DATA_FOLDER, 'user_config.json')
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-dev-key-very-unsafe')
