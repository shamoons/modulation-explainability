#!/bin/bash
# Non-interactive script to generate perturbations in a screen session

echo "Starting perturbation generation in background..."
echo "Command: uv run python src/perturb_constellations.py --percents 1 2 3 4 5 --random"

# Create screen session and run the command (detached)
screen -dmS perturbations bash -c '
    cd /home/shamoon/modulation-explainability
    echo "Starting perturbation generation at $(date)" | tee perturbation_log.txt
    uv run python src/perturb_constellations.py \
        --percents 1 2 3 4 5 \
        --random \
        --source constellation_diagrams \
        --output perturbed_constellations 2>&1 | tee -a perturbation_log.txt
    echo "Perturbation generation completed at $(date)" | tee -a perturbation_log.txt
'

echo "✓ Perturbation generation started in screen session 'perturbations'"
echo ""
echo "Commands:"
echo "  Check progress:  screen -r perturbations"
echo "  List screens:    screen -ls"
echo "  Detach:          Ctrl-A, then D"
echo "  View log:        tail -f perturbation_log.txt"
echo ""
echo "The process is now running in the background."