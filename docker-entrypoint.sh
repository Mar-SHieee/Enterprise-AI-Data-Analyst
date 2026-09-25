#!/usr/bin/env bash

# Build the artifacts the app needs only if they are missing.
# This keeps restarts fast and never overwrites existing data/artifacts.
set -euo pipefail

DB_PATH="${DB_PATH:-data/retail.db}"

# --------------------------------------------------------------------------
# Build SQLite warehouse if it does not exist
# --------------------------------------------------------------------------
if [ ! -f "$DB_PATH" ]; then
  echo "[entrypoint] Database not found. Building warehouse at: $DB_PATH"

  if ! python scripts/build_database.py --out "$DB_PATH"; then
    echo "[entrypoint] ERROR: Failed to build the SQLite warehouse."
    echo "[entrypoint] The application cannot start without the database."
    exit 1
  fi

  echo "[entrypoint] Database built successfully."
else
  echo "[entrypoint] Database already exists: $DB_PATH"
fi

# --------------------------------------------------------------------------
# Train model artifacts if they do not exist
# --------------------------------------------------------------------------
MODEL_PATH="models/classical_repeat_purchase_rf_v1.joblib"

if [ ! -f "$MODEL_PATH" ]; then
  echo "[entrypoint] Model artifacts not found. Training models..."

  if ! python scripts/train_models.py; then
    echo "[entrypoint] WARNING: Model training failed."
    echo "[entrypoint] Prediction features may be unavailable."
  else
    echo "[entrypoint] Model artifacts created successfully."
  fi
else
  echo "[entrypoint] Model artifacts already exist."
fi

# --------------------------------------------------------------------------
# Start application
# --------------------------------------------------------------------------
echo "[entrypoint] Starting application: $*"

exec "$@"
