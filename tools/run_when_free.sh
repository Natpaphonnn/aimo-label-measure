#!/usr/bin/env bash
# Run one resumable Mac job politely (the author 2026-10-25: do not crowd his other projects, but do not starve ours either).
# Loop: wait until the machine has room (power ok, enough free memory, no known heavy job of another project), run the
# command at nice +10 under caffeinate, and if it is stopped (mem_guard SIGTERM -> checkpoint, exit 3; or killed) wait for
# room again and rerun it; the job itself resumes from its checkpoint (trace_states.py .part files, gen_mlx output file).
# Ends when the command exits 0, when DONE_FILE exists, or on runs/STOP.
# Usage: tools/run_when_free.sh DONE_FILE LOG -- command args...
# Env: MIN_FREE_START (default 35, % free memory to start), MIN_BATT_START (default 40, % when on battery),
#      BUSY_PAT (default: CrunchDAO run_daily) = pgrep -f pattern of other projects' heavy jobs to wait for.
cd "$(dirname "$0")/.."
DONE=$1; LOG=$2; shift 3
MIN_FREE_START=${MIN_FREE_START:-35}; MIN_BATT_START=${MIN_BATT_START:-40}; BUSY_PAT=${BUSY_PAT:-CrunchDAO/datacrunch/pipeline/run_daily.py}
room() {
  local free b batt
  free=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
  b=$(pmset -g batt); batt=$(echo "$b" | grep -o '[0-9]*%' | head -1 | tr -d %)
  if echo "$b" | grep -q "Battery Power" && [ "${batt:-100}" -lt "$MIN_BATT_START" ]; then why="battery ${batt}%"; return 1; fi
  if pgrep -f "$BUSY_PAT" >/dev/null; then why="other project running ($BUSY_PAT)"; return 1; fi
  if [ "${free:-0}" -lt "$MIN_FREE_START" ]; then why="free memory ${free}%"; return 1; fi
  return 0
}
n=0
while [ ! -f "$DONE" ]; do
  [ -f runs/STOP ] && { echo "$(date) runs/STOP present; exiting" >> "$LOG"; exit 0; }
  last=""
  until room; do
    [ "$why" != "$last" ] && echo "$(date) waiting: $why" >> "$LOG"; last=$why; sleep 120
    [ -f runs/STOP ] && exit 0
  done
  n=$((n+1)); echo "$(date) start #$n: $*" >> "$LOG"
  caffeinate -i nice -n 10 "$@" >> "$LOG" 2>&1; rc=$?
  echo "$(date) exit $rc" >> "$LOG"
  [ $rc = 0 ] && break
  sleep 300   # let whatever needed the memory get going before checking again
done
echo "$(date) done ($DONE)" >> "$LOG"
