#!/bin/bash
IFACE=eno1
TIMEOUT=60
elapsed=0
while ! ip addr show "$IFACE" 2>/dev/null | grep -q "inet "; do
  sleep 1
  elapsed=$((elapsed + 1))
  if [ "$elapsed" -ge "$TIMEOUT" ]; then
    echo "Timed out waiting for $IFACE to get an IP" >&2
    exit 1
  fi
done
exit 0
