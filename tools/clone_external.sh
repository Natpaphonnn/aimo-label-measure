#!/usr/bin/env bash
# Re-clone the official and third-party repositories consulted by this project (not vendored).
set -euo pipefail
cd "$(dirname "$0")/../external" 2>/dev/null || { mkdir -p "$(dirname "$0")/../external"; cd "$(dirname "$0")/../external"; }
for r in aimo-interp/getting-started aimo-interp/baselines aimo-interp/aimo-interp.github.io \
         byeolki/aimo-interp CVasilopoulos/aimo-bas renxuhangrt-dotcom/aimo-interp-probe-stress-tests \
         coderdailyone/aimo-interp-challenge chrrod/getting-started; do
  d=$(echo "$r" | tr '/' '__'); [ -d "$d" ] || git clone -q "https://github.com/$r.git" "$d"
done
