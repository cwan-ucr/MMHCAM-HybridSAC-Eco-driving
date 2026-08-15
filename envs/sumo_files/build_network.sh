#!/bin/bash
# Build the SUMO network file from components
# Run this once before training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

netconvert \
    --node-files="${SCRIPT_DIR}/intersection.nod.xml" \
    --edge-files="${SCRIPT_DIR}/intersection.edg.xml" \
    --connection-files="${SCRIPT_DIR}/intersection.con.xml" \
    --tllogic-files="${SCRIPT_DIR}/intersection.tll.xml" \
    --output-file="${SCRIPT_DIR}/intersection.net.xml" \
    --no-turnarounds true

echo "Network file generated: ${SCRIPT_DIR}/intersection.net.xml"
