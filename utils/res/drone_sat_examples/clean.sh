#!/usr/bin/env bash

find . -type f \( -name "*.png" -o -name "*.json" \) | grep -v -E "drone_sat_example_(19|21|37|55|82)\.(png|json)$" | xargs rm -f
