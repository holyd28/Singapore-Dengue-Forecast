#!/bin/bash

echo "Building Dengue Forecast Environment..."
docker build -t dengue_forecast .

echo "Running Dengue Forecaster..."
docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/results:/app/results" \
    dengue_forecast