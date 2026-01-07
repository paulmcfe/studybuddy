#!/bin/bash
# Only deploy if changes are in v2-rag-from-scratch
# Exit 1 (true) = proceed with build, Exit 0 (false) = skip build
git diff HEAD^ HEAD --name-only | grep -q "^v2-rag-from-scratch/" && exit 1 || exit 0