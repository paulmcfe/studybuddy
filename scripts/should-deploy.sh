#!/bin/bash
# Deploy if changes are in v3-the-agent-loop or vercel config
# Exit 1 (true) = proceed with build, Exit 0 (false) = skip build
git diff HEAD^ HEAD --name-only | grep -qE "^v3-the-agent-loop/|^vercel\.json" && exit 1 || exit 0
