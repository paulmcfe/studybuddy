#!/bin/bash
# Only deploy if changes are in v4-agentic-rag
# Exit 1 (true) = proceed with build, Exit 0 (false) = skip build
git diff HEAD^ HEAD --name-only | grep -q "^v4-agentic-rag/" && exit 1 || exit 0