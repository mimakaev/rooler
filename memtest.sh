#!/bin/bash
"$@" >/tmp/mt.out 2>&1 &
pid=$!
peak=0
while kill -0 $pid 2>/dev/null; do
  hwm=$(awk '/VmHWM/{print $2}' /proc/$pid/status 2>/dev/null)
  [ -n "$hwm" ] && [ "$hwm" -gt "$peak" ] && peak=$hwm
  sleep 0.3
done
wait $pid; rc=$?
echo "PEAK_RSS_GB=$(awk "BEGIN{printf \"%.2f\", $peak/1048576}") rc=$rc"
