"""
OptiCrop — Application Configuration
"""
import os


class Config:
    """Base configuration class."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "opticrop-secret-key-change-in-production")
    DEBUG = False
    TESTING = False

    # Paths
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # Model file paths
    MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
    ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

    # Dataset path
    DATASET_PATH = os.path.join(DATA_DIR, "crop_recommendation.csv")


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True


# Configuration dictionary
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}
