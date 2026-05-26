import hashlib
import os
from datetime import timedelta

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager, get_jwt_identity, jwt_required
import numpy as np

from .auth import auth_bp
from .database import Dataset, ModelMetadata, PredictionLog, User, db, init_db
from .ml_pipeline import predict_user_document, train_user_model
from .summarizer import summarize_text
from .text_extractor import extract_text_from_bytes, extract_text_from_upload, normalize_extracted_text, save_uploaded_file, validate_upload


def create_app() -> Flask:
    app = Flask(__name__, instance_relative_config=True)

    # Secrets
    app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-me")
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-flask-secret-change-me")

    # Persistence (SQLite)
    db_path = os.environ.get("DATABASE_PATH", os.path.join(app.root_path, "classifier.db"))
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # JWT
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=int(os.environ.get("JWT_ACCESS_EXPIRES_DAYS", "1")))

    # Upload
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
    app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "uploads")

    # CORS
    cors_origin = os.environ.get("CORS_ORIGIN")
    if cors_origin:
        CORS(app, resources={r"/*": {"origins": cors_origin}}, supports_credentials=True)
    else:
        # Dev-friendly default
        CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

    db.init_app(app)
    with app.app_context():
        db.create_all()

    JWTManager(app)

    app.register_blueprint(auth_bp)

    register_routes(app)
    return app


def _user_id_from_jwt() -> int:
    user_id = get_jwt_identity()
    if user_id is None:
        raise ValueError("Missing JWT identity")
    return int(user_id)


def _content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def register_routes(app: Flask):
    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/upload-training")
    @jwt_required()
    def upload_training():
        user_id = _user_id_from_jwt()
        category = (request.form.get("category") or "").strip()
        if not category:
            return jsonify({"error": "Missing category"}), 400

        if "file" not in request.files:
            return jsonify({"error": "Missing file"}), 400

        file_storage = request.files["file"]
        try:
            validate_upload(file_storage, max_bytes=app.config["MAX_CONTENT_LENGTH"])
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        # Read bytes once; also allow saving after extraction.
        file_storage.stream.seek(0)
        file_bytes = file_storage.read()
        file_storage.stream.seek(0)

        ext = os.path.splitext(file_storage.filename or "")[1].lower()
        try:
            extracted = extract_text_from_bytes(file_bytes, ext=ext)
            extracted = normalize_extracted_text(extracted)
        except Exception as e:
            return jsonify({"error": f"Failed to extract text: {str(e)}"}), 400

        if not extracted or not extracted.strip():
            return jsonify({"error": "Document contained no extractable text"}), 400

        saved_path = None
        try:
            saved_path = save_uploaded_file(file_storage, app.config["UPLOAD_FOLDER"] + f"/user_{user_id}")
        except Exception:
            saved_path = None

        content_hash = _content_hash(extracted)
        # If identical doc already exists, still store under new id, but this helps duplicates.
        dataset = Dataset(
            user_id=user_id,
            filename=os.path.basename(file_storage.filename or "document"),
            category=category,
            extracted_text=extracted,
            content_hash=content_hash,
        )
        db.session.add(dataset)
        db.session.commit()

        return (
            jsonify(
                {
                    "message": "Training document uploaded",
                    "dataset_id": dataset.id,
                    "filename": dataset.filename,
                    "category": dataset.category,
                    "extracted_text_preview": extracted[:800],
                    "saved_path": saved_path,
                }
            ),
            201,
        )

    @app.post("/extract-text")
    @jwt_required()
    def extract_text():
        if "file" not in request.files:
            return jsonify({"error": "Missing file"}), 400
        file_storage = request.files["file"]
        try:
            validate_upload(file_storage, max_bytes=app.config["MAX_CONTENT_LENGTH"])
            extracted_text = extract_text_from_upload(file_storage)
            extracted_text = normalize_extracted_text(extracted_text)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        if not extracted_text:
            return jsonify({"error": "Document contained no extractable text"}), 400
        return jsonify({"extracted_text": extracted_text, "preview": extracted_text[:800]})

    @app.post("/summarize")
    @jwt_required()
    def summarize():
        payload = request.get_json(silent=True) or {}
        text = payload.get("text") or ""
        max_sentences = payload.get("max_sentences") or 3
        try:
            summary = summarize_text(text, max_sentences=int(max_sentences))
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"summary": summary})

    @app.post("/train-model")
    @jwt_required()
    def train_model():
        user_id = _user_id_from_jwt()
        datasets = Dataset.query.filter_by(user_id=user_id).order_by(Dataset.upload_date.desc()).all()
        if not datasets:
            return jsonify({"error": "No datasets found. Upload training data first."}), 400

        result = train_user_model(app.root_path, user_id, datasets)
        if result.status != "trained":
            return (
                jsonify(
                    {
                        "status": result.status,
                        "accuracy": result.accuracy,
                        "trained_docs": result.trained_docs,
                        "categories": result.categories,
                        "message": "Need at least 2 categories with sufficient documents to train.",
                    }
                ),
                200,
            )

        return jsonify(
            {
                "status": result.status,
                "accuracy": result.accuracy,
                "trained_docs": result.trained_docs,
                "categories": result.categories,
            }
        )

    @app.post("/predict")
    @jwt_required()
    def predict():
        user_id = _user_id_from_jwt()
        if "file" not in request.files:
            return jsonify({"error": "Missing file"}), 400
        file_storage = request.files["file"]
        try:
            validate_upload(file_storage, max_bytes=app.config["MAX_CONTENT_LENGTH"])
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        file_storage.stream.seek(0)
        file_bytes = file_storage.read()
        file_storage.stream.seek(0)

        ext = os.path.splitext(file_storage.filename or "")[1].lower()
        try:
            extracted = extract_text_from_bytes(file_bytes, ext=ext)
            extracted = normalize_extracted_text(extracted)
        except Exception as e:
            return jsonify({"error": f"Failed to extract text: {str(e)}"}), 400

        if not extracted or not extracted.strip():
            return jsonify({"error": "Document contained no extractable text"}), 400

        datasets = Dataset.query.filter_by(user_id=user_id).all()
        try:
            prediction = predict_user_document(app.root_path, user_id, extracted_text=extracted, datasets=datasets)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        # Log prediction for analytics.
        db.session.add(
            PredictionLog(
                user_id=user_id,
                predicted_category=prediction.get("predicted_category"),
                confidence=prediction.get("confidence"),
                duplicate_similarity=prediction.get("duplicate_similarity"),
            )
        )
        db.session.commit()

        return jsonify(
            {
                "predicted_category": prediction.get("predicted_category"),
                "confidence": prediction.get("confidence"),
                "duplicate_similarity": prediction.get("duplicate_similarity"),
                "tags": prediction.get("tags"),
                "keywords": prediction.get("keywords"),
                "extracted_text_preview": extracted[:800],
            }
        )

    @app.get("/datasets")
    @jwt_required()
    def get_datasets():
        user_id = _user_id_from_jwt()
        datasets = Dataset.query.filter_by(user_id=user_id).order_by(Dataset.upload_date.desc()).all()
        return jsonify(
            [
                {
                    "id": d.id,
                    "filename": d.filename,
                    "category": d.category,
                    "extracted_text": d.extracted_text,
                    "upload_date": d.upload_date.isoformat(),
                }
                for d in datasets
            ]
        )

    @app.delete("/dataset/<int:dataset_id>")
    @jwt_required()
    def delete_dataset(dataset_id: int):
        user_id = _user_id_from_jwt()
        d = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()
        if d is None:
            return jsonify({"error": "Dataset not found"}), 404
        db.session.delete(d)
        db.session.commit()
        return jsonify({"message": "Dataset deleted"})

    @app.get("/analytics")
    @jwt_required()
    def analytics():
        user_id = _user_id_from_jwt()

        total_datasets = Dataset.query.filter_by(user_id=user_id).count()
        meta = ModelMetadata.query.filter_by(user_id=user_id).first()
        model_accuracy = meta.accuracy if meta is not None else None

        # Category distribution
        category_rows = (
            Dataset.query.with_entities(Dataset.category, db.func.count(Dataset.id))
            .filter(Dataset.user_id == user_id)
            .group_by(Dataset.category)
            .all()
        )
        category_distribution = {cat: int(cnt) for cat, cnt in category_rows}

        recent_uploads = (
            Dataset.query.filter_by(user_id=user_id)
            .order_by(Dataset.upload_date.desc())
            .limit(5)
            .all()
        )

        prediction_logs = (
            PredictionLog.query.filter_by(user_id=user_id)
            .order_by(PredictionLog.created_at.desc())
            .limit(30)
            .all()
        )
        total_predictions = PredictionLog.query.filter_by(user_id=user_id).count()
        confidences = [p.confidence for p in prediction_logs if p.confidence is not None]
        avg_confidence = float(np.mean(confidences)) if len(confidences) else None
        predicted_cat_rows = (
            PredictionLog.query.with_entities(PredictionLog.predicted_category, db.func.count(PredictionLog.id))
            .filter(PredictionLog.user_id == user_id)
            .group_by(PredictionLog.predicted_category)
            .order_by(db.func.count(PredictionLog.id).desc())
            .limit(6)
            .all()
        )
        top_predicted_categories = {cat: int(cnt) for cat, cnt in predicted_cat_rows if cat}

        return jsonify(
            {
                "total_datasets": total_datasets,
                "model_accuracy": model_accuracy,
                "category_distribution": category_distribution,
                "recent_uploads": [
                    {
                        "id": d.id,
                        "filename": d.filename,
                        "category": d.category,
                        "upload_date": d.upload_date.isoformat(),
                    }
                    for d in recent_uploads
                ],
                "prediction_stats": {
                    "total_predictions": total_predictions,
                    "average_confidence": avg_confidence,
                    "top_predicted_categories": top_predicted_categories,
                },
            }
        )

    @app.errorhandler(413)
    def file_too_large(e):
        return jsonify({"error": "File too large"}), 413

    @app.errorhandler(Exception)
    def unhandled_exception(e):
        # Avoid leaking internals; but keep message for developer UX.
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)

