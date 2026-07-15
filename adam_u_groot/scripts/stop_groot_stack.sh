#!/usr/bin/env bash
# Backward-compatible wrapper — prefer cleanup_groot_stack.sh for --status / --dry-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/cleanup_groot_stack.sh" --stop --wait "$@"
