#!/bin/bash
# Background watcher: every 30s log 409/502 counts, error-file count, sandbox
# process counts, and LLM proxy health. Survives ssh drops (run under nohup).
D=/root/Dressage/log/traj_err/qwen3.5-35B-A3B-sync-local
while true; do
  ts=$(date +%H:%M:%S)
  f409=$(grep -rh "409 Conflict" "$D" 2>/dev/null | wc -l)
  f502=$(grep -rh "502 Bad Gateway" "$D" 2>/dev/null | wc -l)
  files=$(find "$D" -name error.json 2>/dev/null | wc -l)
  bw=$(ps -eo comm | grep -cE '^bwrap$')
  oc=$(ps -eo comm | grep -c opencode)
  proxy=$(curl -s -m2 -o /dev/null -w '%{http_code}' http://10.244.156.90:8800/health 2>/dev/null)
  echo "$ts 409=$f409 502=$f502 err_files=$files bwrap=$bw opencode=$oc proxy=$proxy"
  sleep 30
done
