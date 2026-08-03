#!/usr/bin/env bash
# Doctavox — one-shot provisioning for an Oracle Cloud Ampere A1 VM (Ubuntu).
#
# Installs Docker, opens the host firewall, fetches the repo, writes .env and
# brings the stack up behind HTTPS. Safe to re-run: every step is idempotent,
# so this doubles as the redeploy script after a `git push`.
#
#   bash infra/oracle/bootstrap.sh
#
# Configuration comes from the environment (all optional):
#   GROQ_API_KEY                 AI tuning; the API degrades gracefully without it
#   APP_FOOTBALL_DATA_API_TOKEN  /api/chelsea/football
#   APP_CORS_ORIGINS             comma-separated browser origins (default: *)
#   API_DOMAIN                   TLS hostname (default: <dashed-public-ip>.sslip.io)
#   REPO_URL / BRANCH            where to clone from (default: origin of this repo)
#   GITHUB_TOKEN                 only if the repo is private

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/anubhavmalhotra0395/Resume-2026.git}"
BRANCH="${BRANCH:-main}"
APP_DIR="${APP_DIR:-$HOME/doctavox}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mwarning: %s\033[0m\n' "$*" >&2; }
die() { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -ne 0 ] || die "run as the default user (ubuntu), not root — it needs sudo, not a root shell"
sudo -n true 2>/dev/null || sudo true || die "passwordless sudo unavailable"

# ---------------------------------------------------------------- public IP
# Oracle VMs don't hold the public IP on the NIC (it's 1:1 NAT), so ask outside.
PUBLIC_IP="${PUBLIC_IP:-$(curl -fsS --max-time 10 https://api.ipify.org || true)}"
[ -n "$PUBLIC_IP" ] || die "could not determine the public IP; set PUBLIC_IP=... and re-run"
API_DOMAIN="${API_DOMAIN:-${PUBLIC_IP//./-}.sslip.io}"
log "public IP $PUBLIC_IP  →  https://$API_DOMAIN"

# ------------------------------------------------------------------- docker
if ! command -v docker >/dev/null 2>&1; then
  log "installing Docker"
  sudo apt-get update -qq
  sudo apt-get install -y -qq ca-certificates curl git
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker "$USER"
else
  log "Docker already installed ($(docker --version))"
fi

# The docker group only applies to new logins, so this run keeps using sudo.
DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

# ----------------------------------------------------------------- firewall
# Oracle's Ubuntu images ship an iptables INPUT chain that REJECTs everything
# except SSH. Opening the port in the OCI security list is NOT enough — this
# is the single most common reason a fresh Oracle VM "ignores" port 443.
log "opening ports 80/443 on the host firewall"
sudo apt-get install -y -qq iptables-persistent >/dev/null 2>&1 || true
for port in 80 443; do
  if ! sudo iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null; then
    sudo iptables -I INPUT 1 -p tcp --dport "$port" -j ACCEPT
  fi
done
sudo netfilter-persistent save >/dev/null 2>&1 || warn "could not persist iptables rules (they will reset on reboot)"
if sudo ufw status 2>/dev/null | grep -q "Status: active"; then
  sudo ufw allow 80/tcp >/dev/null && sudo ufw allow 443/tcp >/dev/null
fi

# --------------------------------------------------------------------- code
if [ -d "$APP_DIR/.git" ]; then
  log "updating $APP_DIR"
  git -C "$APP_DIR" fetch --depth 1 origin "$BRANCH"
  git -C "$APP_DIR" reset --hard "origin/$BRANCH"
else
  log "cloning $REPO_URL"
  CLONE_URL="$REPO_URL"
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    CLONE_URL="${REPO_URL/https:\/\//https://$GITHUB_TOKEN@}"
  fi
  git clone --depth 1 --branch "$BRANCH" "$CLONE_URL" "$APP_DIR" \
    || die "clone failed — if the repo is private, re-run with GITHUB_TOKEN=<pat>"
  # Don't leave the token sitting in .git/config.
  git -C "$APP_DIR" remote set-url origin "$REPO_URL"
fi

cd "$APP_DIR/infra/oracle"

# ---------------------------------------------------------------------- env
# Written fresh each run from the environment, but existing values are kept
# when the corresponding variable isn't supplied — so a redeploy without
# secrets in the environment doesn't wipe the secrets already on disk.
keep() { [ -f .env ] && sed -n "s/^$1=//p" .env | head -1 || true; }
GROQ_API_KEY="${GROQ_API_KEY:-$(keep GROQ_API_KEY)}"
APP_FOOTBALL_DATA_API_TOKEN="${APP_FOOTBALL_DATA_API_TOKEN:-$(keep APP_FOOTBALL_DATA_API_TOKEN)}"
APP_CORS_ORIGINS="${APP_CORS_ORIGINS:-$(keep APP_CORS_ORIGINS)}"
APP_CORS_ORIGINS="${APP_CORS_ORIGINS:-*}"

log "writing .env"
umask 077
cat > .env <<EOF
# Generated by infra/oracle/bootstrap.sh — not tracked by git.
API_DOMAIN=$API_DOMAIN
APP_CORS_ORIGINS=$APP_CORS_ORIGINS
GROQ_API_KEY=$GROQ_API_KEY
APP_FOOTBALL_DATA_API_TOKEN=$APP_FOOTBALL_DATA_API_TOKEN
EOF
umask 022
[ -n "$GROQ_API_KEY" ] || warn "GROQ_API_KEY is empty — AI tuning will fall back to heuristics"

# ------------------------------------------------------------------- deploy
log "building images (first run pulls PyTorch — expect 10-20 minutes)"
$DOCKER compose build

log "starting stack"
$DOCKER compose up -d --remove-orphans

# ------------------------------------------------------------------- verify
log "waiting for the API to answer"
for i in $(seq 1 60); do
  if curl -fsS --max-time 5 http://127.0.0.1:8000/healthz >/dev/null 2>&1; then
    echo "  API healthy after ${i}0s"
    break
  fi
  [ "$i" -lt 60 ] || die "API never became healthy — check: $DOCKER compose logs api"
  sleep 10
done

log "waiting for the Let's Encrypt certificate"
for i in $(seq 1 30); do
  if curl -fsS --max-time 10 "https://$API_DOMAIN/healthz" >/dev/null 2>&1; then
    printf '\n\033[1;32mDeployed: https://%s\033[0m\n\n' "$API_DOMAIN"
    exit 0
  fi
  sleep 10
done

warn "HTTPS not answering yet. Certificate issuance needs port 80 reachable from"
warn "the internet — confirm the OCI security list has ingress rules for 80 and 443."
warn "Check progress with: $DOCKER compose logs caddy"
exit 1
