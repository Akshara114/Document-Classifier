# Train a Document Classifier 

Web application to upload categorized documents, train a document classifier, predict categories, detect duplicates, generate tags, and summarize documents.

## Tech

- Frontend: React (Vite) + Tailwind CSS + Framer Motion + React Router + Axios
- Backend: Flask + Flask-CORS + Flask-JWT-Extended + SQLAlchemy + SQLite
- ML: scikit-learn (TF-IDF + Logistic Regression) + joblib
- Document extraction: PyPDF2, python-docx, pytesseract OCR, Pillow
- Summarization: sumy (TextRank) + nltk tokenization

## Local Development

### 1) Backend

From the repository root:

```bash
cd project/backend
pip install -r requirements.txt
```

Install OCR engine (required for OCR of images):
- `tesseract` must be installed and available on your `PATH`.

Run the backend (choose one):

**From `project/` (recommended):**

```bash
cd project
python -m backend.app
```

**From `project/backend/` (same app, easier on Windows):**

```bash
cd project/backend
python run.py
```

Backend runs on `http://localhost:5000`.

### 2) Frontend

In another terminal:

```bash
cd project/frontend
npm install
npm run dev
```

If `npm install` fails with `UNABLE_TO_VERIFY_LEAF_SIGNATURE` (common behind corporate proxies), use:

```bash
# PowerShell
$env:NODE_OPTIONS='--use-system-ca'
npm install
```

Or set it once for your user: `setx NODE_OPTIONS "--use-system-ca"` (restart the terminal after).

Frontend runs on `http://localhost:5173`.

### 3) Environment variables (optional)

Create environment variables before running:

- `JWT_SECRET_KEY` (default: `dev-secret-change-me`)
- `FLASK_SECRET_KEY` (default: `dev-flask-secret-change-me`)
- `DATABASE_PATH` (default: `backend/classifier.db`)
- `CORS_ORIGIN` (default: allow `*` in dev)
- `MAX_UPLOAD_BYTES` (default: `20*1024*1024`)

## API (Backend)

- `POST /register`
- `POST /login`
- `GET /me`
- `POST /upload-training` (multipart `file`, `category`)
- `POST /extract-text` (multipart `file`)
- `POST /summarize` (json `text`, optional `max_sentences`)
- `POST /train-model`
- `POST /predict` (multipart `file`)
- `GET /datasets`
- `DELETE /dataset/<id>`
- `GET /analytics`

## Production Deployment 

### Backend

Use a WSGI server:

```bash
gunicorn -w 2 -b 0.0.0.0:5000 "backend.app:app"
```

Make sure to set strong secrets:

- `JWT_SECRET_KEY`
- `FLASK_SECRET_KEY`

### Frontend

Build and serve static assets:

```bash
cd project/frontend
npm run build
```

Serve the `dist/` output using your preferred static hosting.

