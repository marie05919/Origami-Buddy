# 🖼️ Folder Classifier — Streamlit

Upload an image or use your camera to classify folder condition with a
fine-tuned model. Classes:

`Crumpled Folder` · `Deformed folder` · `Good folder` · `Inside-Out Folder` · `Ripped folder`

The model runs as a framework-light **TorchScript** artifact, so the app
needs only `torch + torchvision + streamlit` — fast and reproducible anywhere.

```
maria_deploy/
├── app.py              # Streamlit web app (upload + camera)
├── model/
│   ├── model.ts        # TorchScript model the app runs   ← required
│   └── labels.json     # class names + preprocessing spec ← required
├── requirements.txt    # runtime deps (pinned)
├── setup.sh            # one-command venv setup
└── Dockerfile          # fully reproducible container
```

## Quick start

Works with **Python 3.9+**.

```bash
./setup.sh                  # creates .venv + installs pinned deps
source .venv/bin/activate
streamlit run app.py        # open http://localhost:8501
```

Then use the **📁 Upload** or **📷 Camera** tab and pick/take a photo — the
top-5 class probabilities are shown.

### Manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Run with Docker

No local Python needed — same environment everywhere.

```bash
docker build -t folder-classifier .
docker run -p 8501:8501 folder-classifier
# open http://localhost:8501
```

## Reproduce on another device

Copy the whole folder (it contains `model/model.ts` and `model/labels.json`),
then run **either** the Quick start or Docker steps above. Nothing else needed.

## Notes

- **CPU inference** — runs anywhere, no GPU required.
- Preprocessing matches the original training pipeline exactly (Resize 256
  center-crop, pixels ÷255, no normalization) — stored in `labels.json`.
- **Camera** needs a browser with camera access; over a network use HTTPS or
  `localhost` (browsers block the camera on plain HTTP otherwise).
