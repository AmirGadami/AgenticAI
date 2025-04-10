#!/bin/bash

echo "🔧 Running post-setup tasks..."

# Set PYTHONPATH to current project root
export PYTHONPATH=$(pwd)
echo "✅ PYTHONPATH set to: $PYTHONPATH"

# Run your script right here, with PYTHONPATH at the top
python -c "import sys; print(sys.path)"  # Debug: shows it's working
# python agents/frontier_agent.py  # 👈 run your actual script here

echo "✅ Post-setup complete!"