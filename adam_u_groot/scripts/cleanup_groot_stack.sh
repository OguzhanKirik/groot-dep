#!/usr/bin/env bash
# Stop GR00T policy servers, Isaac eval_groot runs, and related conda wrappers.
#
# Usage:
#   bash cleanup_groot_stack.sh              # stop (default)
#   bash cleanup_groot_stack.sh --status     # show what is running
#   bash cleanup_groot_stack.sh --dry-run    # print actions only
#   bash cleanup_groot_stack.sh --wait       # stop, then wait for port/GPU settle
#
# Environment:
#   GROOT_PORT   ZMQ port (default 5555)
#   GRACE_SEC    SIGTERM grace period before SIGKILL (default 8)

set -euo pipefail

GROOT_PORT="${GROOT_PORT:-5555}"
GRACE_SEC="${GRACE_SEC:-8}"
QUIET="${QUIET:-0}"

# Match our stack only — avoid killing unrelated python jobs.
PATTERNS=(
  "run_gr00t_server.py"
  "eval_groot.py"
  "conda run.*run_gr00t_server.py"
  "conda run.*eval_groot.py"
)

log() {
  [[ "${QUIET}" == "1" ]] && return 0
  echo "$@"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Stop stale GR00T / Isaac eval processes and free port ${GROOT_PORT}.

Options:
  --stop       Stop matching processes (default)
  --status     List matching processes and port ${GROOT_PORT} without killing
  --dry-run    Show what would be stopped
  --wait       After stop, wait until port ${GROOT_PORT} is free (up to 30s)
  --quiet      Minimal output (for sourcing from other scripts)
  -h, --help   Show this help

Examples:
  bash $(basename "$0") --status
  bash $(basename "$0") --wait
  GROOT_PORT=5556 bash $(basename "$0")
EOF
}

collect_pids() {
  local seen="" pid pattern
  for pattern in "${PATTERNS[@]}"; do
    while IFS= read -r pid; do
      [[ -z "${pid}" ]] && continue
      # Skip our own pgrep/bash invocations.
      if [[ " ${seen} " == *" ${pid} "* ]]; then
        continue
      fi
      local cmd
      cmd="$(ps -p "${pid}" -o cmd= 2>/dev/null || true)"
      [[ -z "${cmd}" ]] && continue
      if [[ "${cmd}" == *"pgrep -f"* ]] || [[ "${cmd}" == *"cleanup_groot_stack.sh"* ]]; then
        continue
      fi
      seen="${seen} ${pid}"
      echo "${pid}"
    done < <(pgrep -f "${pattern}" 2>/dev/null || true)
  done
}

port_in_use() {
  if command -v ss >/dev/null 2>&1; then
    ss -tln 2>/dev/null | grep -q ":${GROOT_PORT} "
    return $?
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"${GROOT_PORT}" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  return 1
}

wait_port_free() {
  local i
  for ((i = 1; i <= 30; i++)); do
    if ! port_in_use; then
      log "[cleanup] Port ${GROOT_PORT} is free."
      return 0
    fi
    sleep 1
  done
  log "[cleanup] WARNING: port ${GROOT_PORT} still in use after 30s." >&2
  return 1
}

kill_tree() {
  local pid="$1"
  local signal="$2"
  local child
  for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
    kill_tree "${child}" "${signal}"
  done
  kill "-${signal}" "${pid}" 2>/dev/null || true
}

stop_pids() {
  local dry_run="$1"
  shift
  local pids=("$@")
  if [[ ${#pids[@]} -eq 0 ]]; then
    log "[cleanup] No matching GR00T/Isaac eval processes found."
    return 0
  fi

  log "[cleanup] Found ${#pids[@]} process(es):"
  local pid
  for pid in "${pids[@]}"; do
    ps -p "${pid}" -o pid=,etime=,cmd= 2>/dev/null | sed 's/^/[cleanup]   /' || true
  done

  if [[ "${dry_run}" == "1" ]]; then
    log "[cleanup] Dry run — no signals sent."
    return 0
  fi

  log "[cleanup] Sending SIGTERM (grace ${GRACE_SEC}s)..."
  for pid in "${pids[@]}"; do
    kill_tree "${pid}" TERM
  done

  local deadline=$((SECONDS + GRACE_SEC))
  while (( SECONDS < deadline )); do
    local alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=1
        break
      fi
    done
    [[ "${alive}" -eq 0 ]] && break
    sleep 1
  done

  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      log "[cleanup] SIGKILL pid ${pid} (did not exit after SIGTERM)"
      kill_tree "${pid}" KILL
    fi
  done
}

show_status() {
  mapfile -t pids < <(collect_pids)
  if [[ ${#pids[@]} -eq 0 ]]; then
    log "[cleanup] No GR00T server or eval_groot processes running."
  else
    log "[cleanup] Running processes:"
    local pid
    for pid in "${pids[@]}"; do
      ps -p "${pid}" -o pid=,ppid=,etime=,cmd= 2>/dev/null | sed 's/^/[cleanup]   /' || true
    done
  fi

  if port_in_use; then
    log "[cleanup] Port ${GROOT_PORT}: IN USE"
    if command -v ss >/dev/null 2>&1; then
      ss -tlnp 2>/dev/null | grep ":${GROOT_PORT} " | sed 's/^/[cleanup]   /' || true
    fi
  else
    log "[cleanup] Port ${GROOT_PORT}: free"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    log "[cleanup] GPU compute processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
      | grep -E "python|kit|isaac" | sed 's/^/[cleanup]   /' || log "[cleanup]   (none matching python/kit/isaac)"
  fi
}

main() {
  local mode="stop"
  local dry_run=0
  local do_wait=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --stop) mode="stop" ;;
      --status) mode="status" ;;
      --dry-run) dry_run=1; mode="stop" ;;
      --wait) do_wait=1 ;;
      --quiet) QUIET=1 ;;
      -h|--help) usage; exit 0 ;;
      *)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
    shift
  done

  if [[ "${mode}" == "status" ]]; then
    show_status
    exit 0
  fi

  mapfile -t pids < <(collect_pids)
  stop_pids "${dry_run}" "${pids[@]}"

  if [[ "${dry_run}" == "0" ]] && [[ "${do_wait}" == "1" ]]; then
    wait_port_free || true
    sleep 2
  fi

  if [[ "${dry_run}" == "0" ]] && [[ "${QUIET}" != "1" ]]; then
    log "[cleanup] Done. Run with --status to verify."
  fi
}

main "$@"
