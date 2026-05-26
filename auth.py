from datetime import timedelta

from flask import Blueprint, jsonify, request
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required
from werkzeug.security import check_password_hash, generate_password_hash

from .database import User, db

auth_bp = Blueprint("auth", __name__)


def _parse_json(required_fields):
    payload = request.get_json(silent=True) or {}
    missing = [f for f in required_fields if not payload.get(f)]
    if missing:
        return None, jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
    return payload, None, None


@auth_bp.post("/register")
def register():
    payload, error_resp, status = _parse_json(["name", "email", "password"])
    if error_resp:
        return error_resp, status

    name = payload["name"].strip()
    email = payload["email"].strip().lower()
    password = payload["password"]

    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    if User.query.filter_by(email=email).first() is not None:
        return jsonify({"error": "Email already registered"}), 400

    user = User(
        name=name,
        email=email,
        password=generate_password_hash(password),
    )
    db.session.add(user)
    db.session.commit()

    # Issue access token immediately for better UX.
    token = create_access_token(identity=user.id, expires_delta=timedelta(days=1))
    return (
        jsonify(
            {
                "access_token": token,
                "user": {"id": user.id, "name": user.name, "email": user.email},
            }
        ),
        201,
    )


@auth_bp.post("/login")
def login():
    payload, error_resp, status = _parse_json(["email", "password"])
    if error_resp:
        return error_resp, status

    email = payload["email"].strip().lower()
    password = payload["password"]

    user = User.query.filter_by(email=email).first()
    if user is None or not check_password_hash(user.password, password):
        return jsonify({"error": "Invalid email or password"}), 401

    token = create_access_token(identity=user.id, expires_delta=timedelta(days=1))
    return jsonify(
        {"access_token": token, "user": {"id": user.id, "name": user.name, "email": user.email}}
    )


@auth_bp.get("/me")
@jwt_required()
def me():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if user is None:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"user": {"id": user.id, "name": user.name, "email": user.email}})

