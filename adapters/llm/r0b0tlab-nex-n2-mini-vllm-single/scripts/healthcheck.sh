#!/bin/bash
curl -sf http://localhost:${PORT:-8000}/health || exit 1
