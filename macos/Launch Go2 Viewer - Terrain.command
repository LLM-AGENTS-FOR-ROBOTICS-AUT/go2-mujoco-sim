#!/usr/bin/env bash
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/Launch Go2 Viewer.command" --terrain "$@"
