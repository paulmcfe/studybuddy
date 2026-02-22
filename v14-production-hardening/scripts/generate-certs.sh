#!/bin/bash
# Generate self-signed TLS certificates for local development.
#
# Usage:
#   ./scripts/generate-certs.sh
#
# Creates certs/cert.pem and certs/key.pem for use with nginx.
# These are for development only — use proper certificates in production.

set -e

CERT_DIR="$(dirname "$0")/../certs"
mkdir -p "$CERT_DIR"

echo "Generating self-signed TLS certificate for development..."

openssl req -x509 -newkey rsa:2048 \
    -keyout "$CERT_DIR/key.pem" \
    -out "$CERT_DIR/cert.pem" \
    -days 365 \
    -nodes \
    -subj "/CN=localhost/O=StudyBuddy/C=US"

echo ""
echo "Certificates generated:"
echo "  Certificate: $CERT_DIR/cert.pem"
echo "  Private key: $CERT_DIR/key.pem"
echo ""
echo "Add 'certs/' to your .gitignore to avoid committing these files."
