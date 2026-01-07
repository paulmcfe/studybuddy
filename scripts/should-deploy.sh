#!/bin/bash
# Only deploy if changes are in v1-basic-chatbot
# Exit 1 (true) = proceed with build, Exit 0 (false) = skip build
git diff HEAD^ HEAD --name-only | grep -q "^v1-basic-chatbot/" && exit 1 || exit 0