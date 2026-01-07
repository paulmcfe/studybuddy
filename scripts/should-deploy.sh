#!/bin/bash
# Only deploy if changes are in v1-basic-chatbot
git diff HEAD^ HEAD --name-only | grep -q "^v1-basic-chatbot/"