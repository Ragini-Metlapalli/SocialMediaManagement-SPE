#!/bin/bash
# seed-vault.sh
# Purpose: Manually seeds Vault with the Postgres credentials.
# Usage: ./seed-vault.sh [password]

# Default Password if not provided
DB_PASS=${1:-"postgres"}

echo " Seeding Vault with Database Credentials..."
echo "   Password: $DB_PASS"

# Check if Vault is running
if ! kubectl get pod -l app=vault | grep -q "Running"; then
    echo " Error: Vault pod is not running."
    exit 1
fi

# Execute the Put Command inside the Pod (Using Root Token)
kubectl exec deployment/vault -- vault kv put secret/postgres \
    username="postgres" \
    password="$DB_PASS" \
    host="postgres" \
    port="5432" \
    dbname="prediction_db"

echo " Vault Seeded Successfully!"
echo "   Path: secret/postgres"
echo "   Data: username=postgres, password=***, host=postgres..."
