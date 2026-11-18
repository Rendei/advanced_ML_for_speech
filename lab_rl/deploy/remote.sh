#!/bin/sh
# Runs ON THE SERVER from /opt/c4rl (started by deploy.ps1, or by hand: `sh deploy/remote.sh`).
#   INSTALL_DOCKER=1  install Docker with the official convenience script if it is missing
#   FREE_PORTS=1      stop whatever listens on ports 80/443 (nginx/apache/other containers) - only with your consent
#   SKIP_BUILD=1      do not build on the server (small VPS): the c4rl:prod image is already loaded
set -eu
cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
  if [ "${INSTALL_DOCKER:-0}" = "1" ]; then
    echo "== installing Docker (https://get.docker.com)"
    curl -fsSL https://get.docker.com | sh
  else
    echo "Docker is not installed. Re-run with INSTALL_DOCKER=1 or install it yourself." >&2
    exit 1
  fi
fi

if ! docker compose version >/dev/null 2>&1; then
  if [ "${INSTALL_DOCKER:-0}" = "1" ]; then
    echo "== installing the docker compose plugin"
    apt-get update -qq && apt-get install -y --no-install-recommends docker-compose-v2
  else
    echo "The docker compose plugin is missing. Re-run with INSTALL_DOCKER=1 or install docker-compose-v2." >&2
    exit 1
  fi
fi

echo "== who listens on 80/443 now"
ss -ltnp 2>/dev/null | grep -E ':(80|443)[[:space:]]' || echo "nobody"

if [ "${FREE_PORTS:-0}" = "1" ]; then
  echo "== freeing ports 80/443"
  for svc in nginx apache2 httpd caddy lighttpd; do
    if systemctl is-active --quiet "$svc" 2>/dev/null; then
      echo "stopping and disabling $svc"
      systemctl disable --now "$svc"
    fi
  done
  # containers that publish 80/443 (not ours)
  for id in $(docker ps -q --filter publish=80 --filter publish=443 2>/dev/null); do
    case "$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' "$id")" in
      c4rl-prod) ;;
      *) echo "stopping container $(docker inspect -f '{{.Name}}' "$id")"; docker stop "$id" ;;
    esac
  done
fi

if ss -ltn 2>/dev/null | grep -E ':(80|443)[[:space:]]' >/dev/null && ! docker ps --format '{{.Names}}' | grep -q '^c4rl-prod-caddy'; then
  echo "Ports 80/443 are still busy - free them (FREE_PORTS=1) and run again." >&2
  exit 2
fi

echo "== building and starting"
if [ "${SKIP_BUILD:-0}" = "1" ]; then
  docker compose -f docker-compose.prod.yml up -d --no-build   # image was built elsewhere and loaded with docker load
else
  docker compose -f docker-compose.prod.yml up -d --build
fi

echo "== waiting for the game server to become healthy"
i=0
until [ "$(docker inspect -f '{{.State.Health.Status}}' c4rl-prod-play-1 2>/dev/null)" = "healthy" ]; do
  i=$((i + 1)); [ "$i" -gt 60 ] && { echo "play service did not become healthy" >&2; docker compose -f docker-compose.prod.yml logs --tail 40 play; exit 3; }
  sleep 2
done
echo "play: healthy"

if command -v ufw >/dev/null 2>&1 && ufw status | grep -q "Status: active"; then
  echo "== ufw is active: opening 80/tcp and 443/tcp+udp"
  ufw allow 80/tcp && ufw allow 443/tcp && ufw allow 443/udp
fi

echo "== caddy log (certificate issuance can take up to a minute)"
sleep 20
docker compose -f docker-compose.prod.yml logs --tail 25 caddy
echo
echo "Done. Open https://rendei-cool.ru"
