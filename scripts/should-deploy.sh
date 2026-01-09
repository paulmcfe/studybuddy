#!/bin/bash
# Deploy if changes are in v5-multi-agent or vercel config
# Exit 1 (true) = proceed with build, Exit 0 (false) = skip build
git diff HEAD^ HEAD --name-only | grep -qE "^v5-multi-agent/|^vercel\.json" && exit 1 || exit 0