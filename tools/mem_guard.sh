#!/usr/bin/env bash
# Memory guard for long Mac jobs (2026-10-22: a trace_states.py process grew to a 20 GB footprint with swap 16.8/17.4 GB used,
# the likely cause of the earlier forced quit). Every 30 s: if any of OUR model processes (gen_mlx / trace_states /
# force_at_cuts) has a memory footprint above LIMIT_GB (default 14) or the system memory-free percentage falls below
# MIN_FREE (default 10), that process gets SIGTERM (then SIGKILL after 90 s: trace_states.py saves a checkpoint on SIGTERM after its current record) and the event is logged. gen_mlx resumes from its
# output file, so the night launcher simply relaunches it later. Also stops them on battery below MIN_BATT percent (default 20;
# the author 2026-10-22: battery runs are allowed). Usage: tools/mem_guard.sh [LIMIT_GB] [MIN_FREE] [MIN_BATT] (run in background)
LIMIT_GB=${1:-14}; MIN_FREE=${2:-10}; MIN_BATT=${3:-20}; PAT="src/selflabel/gen_mlx.py|src/features/trace_states.py|src/selflabel/force_at_cuts"
while true; do
  free=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
  b=$(pmset -g batt); batt=$(echo "$b" | grep -o '[0-9]*%' | head -1 | tr -d %); onbatt=0; echo "$b" | grep -q "Battery Power" && onbatt=1
  lowb=0; [ "$onbatt" = 1 ] && [ "${batt:-100}" -lt "$MIN_BATT" ] && lowb=1
  for pid in $(pgrep -f "$PAT"); do
    mem=$(top -l 1 -pid "$pid" -stats mem 2>/dev/null | tail -1)   # e.g. 9420M or 20G
    gb=$(echo "$mem" | awk '/G/{sub("G","");print $1} /M/{sub("M","");print $1/1024} /K/{print 0}')
    over=$(awk -v g="${gb:-0}" -v l="$LIMIT_GB" 'BEGIN{print (g>l)?1:0}')
    low=$(awk -v f="${free:-100}" -v m="$MIN_FREE" 'BEGIN{print (f<m)?1:0}')
    if [ "$over" = 1 ] || [ "$low" = 1 ] || [ "$lowb" = 1 ]; then
      echo "$(date) mem_guard: pid $pid footprint $mem free ${free}% battery ${batt}% (on battery $onbatt) -> SIGTERM ($(ps -o command= -p $pid | cut -c1-80))"
      kill -TERM "$pid"; for _ in 1 2 3 4 5 6 7 8 9; do sleep 10; kill -0 "$pid" 2>/dev/null || break; done; kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"
    fi
  done
  sleep 30
done
