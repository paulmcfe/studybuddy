#!/bin/bash
# Only deploy if changes are in v3-the-agent-loop
# Exit 1 (true) = proceed with build, Exit 0 (false) = skip build
git diff HEAD^ HEAD --name-only | grep -q "^v3-the-agent-loop/" && exit 1 || exit 0