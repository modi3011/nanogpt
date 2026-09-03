#!/usr/bin/env bash
# ==============================================================================
# nanogpt — redeployable Azure Web App (single container) deploy script
#
# One FastAPI container (API + frontend, no DB) built from docker/Dockerfile.prod
# at the repo root, pushed to ACR, and pointed at an Azure Web App for Containers.
# HuggingFace checkpoints are pulled by the container itself at startup, so
# there's no database or persistent volume to manage.
#
# Usage:
#   ./deploy.sh              # deploy (default): build, push, point web app, restart, verify
#   ./deploy.sh provision    # one-time: create RG / ACR / plan / web app if they don't exist yet
#   ./deploy.sh settings     # just refresh app settings (HF_* vars etc.) + restart, no rebuild
#
# Prerequisites:
#   - az CLI, Docker Desktop + buildx installed, already `az login`-ed with the
#     right subscription selected
#   - a deploy.env file next to this script — copy deploy.env.example to
#     deploy.env and fill in your values (deploy.env should be git-ignored,
#     since it holds HF_TOKEN)
#
# `provision` is idempotent — safe to re-run any time, it only creates what's
# missing. `deploy` never creates resources, only redeploys onto what
# `provision` set up.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/deploy.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE — copy deploy.env.example to deploy.env and fill in your values."
  exit 1
fi

set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"

require_var() {
  if [[ -z "${!1:-}" ]]; then
    echo "Missing required config: $1 (set it in deploy.env)"
    exit 1
  fi
}

log() { echo -e "\n\033[1;34m==>\033[0m $1"; }

# ------------------------------------------------------------------------------
# Prerequisites
# ------------------------------------------------------------------------------

check_tools() {
  log "Checking required tools and Azure login"
  az version >/dev/null
  docker --version >/dev/null
  docker buildx version >/dev/null

  if ! az account show >/dev/null 2>&1; then
    echo "Not logged in to Azure. Run 'az login' first, then re-run this script."
    exit 1
  fi
}

# ------------------------------------------------------------------------------
# One-time provisioning — safe to re-run; only creates what's missing.
# ------------------------------------------------------------------------------

provision() {
  require_var RESOURCE_GROUP
  require_var LOCATION
  require_var ACR_NAME
  require_var PLAN_NAME
  require_var WEBAPP_NAME

  log "Ensuring resource group $RESOURCE_GROUP exists"
  az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

  if az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
    log "ACR $ACR_NAME already exists — skipping"
  else
    log "Creating ACR $ACR_NAME"
    az acr create \
      --resource-group "$RESOURCE_GROUP" \
      --name "$ACR_NAME" \
      --sku "${ACR_SKU:-Basic}" \
      --admin-enabled true \
      --output none
  fi
  # Make sure admin creds are on even if the ACR pre-existed without them
  az acr update --name "$ACR_NAME" --admin-enabled true --output none

  if az appservice plan show --name "$PLAN_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
    log "App Service plan $PLAN_NAME already exists — skipping"
  else
    log "Creating Linux App Service plan $PLAN_NAME (${PLAN_SKU:-B1})"
    az appservice plan create \
      --name "$PLAN_NAME" \
      --resource-group "$RESOURCE_GROUP" \
      --is-linux \
      --sku "${PLAN_SKU:-B1}" \
      --output none
  fi

  if az webapp show --name "$WEBAPP_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
    log "Web App $WEBAPP_NAME already exists — skipping"
  else
    log "Creating Web App $WEBAPP_NAME (placeholder image — 'deploy' will point it at the real one)"
    az webapp create \
      --resource-group "$RESOURCE_GROUP" \
      --plan "$PLAN_NAME" \
      --name "$WEBAPP_NAME" \
      --deployment-container-image-name "mcr.microsoft.com/appsvc/staticsite:latest" \
      --output none
  fi

  echo "Provisioning done. Run './deploy.sh' to build and deploy the real image."
}

# ------------------------------------------------------------------------------
# Build + push + point + restart + verify
# ------------------------------------------------------------------------------

acr_login() {
  ACR_LOGIN_SERVER="$(az acr show \
    --name "$ACR_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query loginServer \
    --output tsv)"
  ACR_USERNAME="$(az acr credential show --name "$ACR_NAME" --query "username" --output tsv)"
  ACR_PASSWORD="$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" --output tsv)"
  az acr login --name "$ACR_NAME"
}

apply_app_settings() {
  require_var HF_REPO_STORIES
  require_var HF_REPO_CODE

  log "Refreshing app settings on $WEBAPP_NAME"
  az webapp config appsettings set \
    --name "$WEBAPP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --settings \
      HF_TOKEN="${HF_TOKEN:-}" \
      HF_REPO_STORIES="$HF_REPO_STORIES" \
      HF_REPO_CODE="$HF_REPO_CODE" \
      STORIES_CKPT_PATH="${STORIES_CKPT_PATH:-checkpoints/stories/ckpt.pt}" \
      CODE_CKPT_PATH="${CODE_CKPT_PATH:-checkpoints/code/ckpt.pt}" \
      WEBSITES_PORT=8000 \
      WEBSITES_CONTAINER_START_TIME_LIMIT="${CONTAINER_START_TIME_LIMIT:-600}" \
    --output none
}

verify() {
  local webapp_url
  webapp_url="https://$(az webapp show \
    --name "$WEBAPP_NAME" --resource-group "$RESOURCE_GROUP" \
    --query defaultHostName --output tsv)"

  log "Deployed: $webapp_url"
  log "Waiting for the container to come up (HF checkpoint downloads take longer than a normal cold start)"

  local i
  for i in $(seq 1 12); do
    if curl -sf -o /dev/null "$webapp_url/docs"; then
      echo "App is responding at $webapp_url"
      return 0
    fi
    echo "  not up yet (attempt $i/12) — waiting 15s"
    sleep 15
  done

  echo "Warning: app not responding after 3 minutes — check:"
  echo "  az webapp log tail --name $WEBAPP_NAME --resource-group $RESOURCE_GROUP"
}

deploy() {
  require_var RESOURCE_GROUP
  require_var SRC_DIR
  require_var DOCKERFILE_PATH
  require_var ACR_NAME
  require_var WEBAPP_NAME
  require_var REPO_NAME

  acr_login

  local image="$ACR_LOGIN_SERVER/$REPO_NAME:$TAG"

  log "Building image: $image"
  docker buildx build \
    --platform linux/amd64 \
    --file "$SRC_DIR/$DOCKERFILE_PATH" \
    --tag "$image" \
    --load \
    "$SRC_DIR"
  docker push "$image"

  log "Pointing $WEBAPP_NAME at the new image"
  az webapp config container set \
    --name "$WEBAPP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --container-image-name "$image" \
    --container-registry-url "https://$ACR_LOGIN_SERVER" \
    --container-registry-user "$ACR_USERNAME" \
    --container-registry-password "$ACR_PASSWORD"

  apply_app_settings

  log "Restarting $WEBAPP_NAME"
  az webapp restart --name "$WEBAPP_NAME" --resource-group "$RESOURCE_GROUP"

  verify
}

settings_only() {
  require_var RESOURCE_GROUP
  require_var WEBAPP_NAME

  apply_app_settings

  log "Restarting $WEBAPP_NAME"
  az webapp restart --name "$WEBAPP_NAME" --resource-group "$RESOURCE_GROUP"

  verify
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

TARGET="${1:-deploy}"
check_tools

case "$TARGET" in
  provision) provision ;;
  deploy)    deploy ;;
  settings)  settings_only ;;
  *)
    echo "Unknown target: $TARGET (use: deploy | provision | settings)"
    exit 1
    ;;
esac