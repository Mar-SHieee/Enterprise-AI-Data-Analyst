#!/usr/bin/env bash
# Build the artifacts the app needs, but only if they are missing, so a restart
# is fast and a mounted volume with real data is never overwritten.
set -euo pipefail

DB_PATH="${DB_PATH:-data/retail.db}"

if [ ! -f "$DB_PATH" ]; then
  echo "[entrypoint] building warehouse at $DB_PATH ..."
  python scripts/build_database.py --out "$DB_PATH" || \
    echo "[entrypoint] WARNING: warehouse build failed; the app will start without SQL."
fi

if [ ! -f "models/classical_repeat_purchase_rf_v1.joblib" ]; then
  echo "[entrypoint] training model artifacts ..."
  python scripts/train_models.py || \
    echo "[entrypoint] WARNING: training failed; predictions will be disabled."
fi

echo "[entrypoint] starting: $*"
exec "$@"
