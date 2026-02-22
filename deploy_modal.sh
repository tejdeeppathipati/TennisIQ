#!/usr/bin/env bash
# Deploy the tennisiq-court app to Modal so the pipeline can find it.
# Run from TennisIQ directory: ./deploy_modal.sh
# Requires: pip install modal, modal setup (or token set)
set -e
cd "$(dirname "$0")"
echo "Deploying tennisiq-court to Modal (from $(pwd))..."
python -m modal deploy tennisiq/modal_court.py
echo "Done. App 'tennisiq-court' should now be available in environment 'main'."
