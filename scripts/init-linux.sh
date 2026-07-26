#!/usr/bin/env sh
set -eu

UID_VALUE="${APP_UID:-10001}"
GID_VALUE="${APP_GID:-10001}"

mkdir -p workspace
printf '%s\n' "Add the provider keys you use to .env before starting."

if command -v sudo >/dev/null 2>&1; then
  sudo chown -R "$UID_VALUE:$GID_VALUE" workspace
else
  chown -R "$UID_VALUE:$GID_VALUE" workspace
fi

printf '%s\n' "Workspace is writable by $UID_VALUE:$GID_VALUE."
