#!/bin/bash
# Deploy if changes are in v8-evaluation, vercel config, or scripts
# Exit 1 (true) = proceed with build, Exit 0 (false) = skip build
git diff HEAD^ HEAD --name-only | grep -qE "^v8-evaluation/|^vercel\.json|^scripts/" && exit 1 || exit 0
