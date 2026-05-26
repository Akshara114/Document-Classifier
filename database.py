import os
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def _utcnow():
    return datetime.utcnow()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(190), unique=True, nullable=False, index=True)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=_utcnow)

    datasets = db.relationship("Dataset", backref="user", lazy=True, cascade="all, delete-orphan")
    prediction_logs = db.relationship(
        "PredictionLog", backref="user", lazy=True, cascade="all, delete-orphan"
    )


class Dataset(db.Model):
    __tablename__ = "datasets"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(120), nullable=False, index=True)
    extracted_text = db.Column(db.Text, nullable=False)
    upload_date = db.Column(db.DateTime, nullable=False, default=_utcnow)
    content_hash = db.Column(db.String(64), nullable=True, index=True)


class ModelMetadata(db.Model):
    """
    Per-user model stats and paths. Keeping this in DB makes analytics independent of filesystem state.
    """

    __tablename__ = "model_metadata"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True, index=True)
    model_path = db.Column(db.String(500), nullable=False)
    vectorizer_path = db.Column(db.String(500), nullable=False)
    labels = db.Column(db.Text, nullable=False)  # JSON list
    accuracy = db.Column(db.Float, nullable=True)
    trained_docs = db.Column(db.Integer, nullable=True)
    categories = db.Column(db.Text, nullable=True)  # JSON list
    last_trained_at = db.Column(db.DateTime, nullable=False, default=_utcnow)


class PredictionLog(db.Model):
    __tablename__ = "prediction_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    predicted_category = db.Column(db.String(120), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    duplicate_similarity = db.Column(db.Float, nullable=True)  # 0-100
    created_at = db.Column(db.DateTime, nullable=False, default=_utcnow)


def init_db(app):
    os.makedirs(os.path.join(app.root_path, "models"), exist_ok=True)
    db.init_app(app)
    with app.app_context():
        db.create_all()

