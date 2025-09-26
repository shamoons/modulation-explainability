#!/bin/bash
# Script to run perturbation evaluation in a screen session

echo "Starting perturbation evaluation in screen session..."
echo "This will evaluate 15 different perturbation types on the test set."
echo "Estimated runtime: 2-3 hours"
echo ""

# Create screen session and run the command (detached)
screen -dmS perturbation_eval bash -c '
    cd /home/shamoon/modulation-explainability
    echo "Starting perturbation evaluation at $(date)" | tee perturbation_evaluation_log.txt
    echo "================================================" | tee -a perturbation_evaluation_log.txt
    
    uv run python papers/ELSP_Paper/scripts/evaluate_perturbations.py \
        --checkpoint checkpoints/best_model_resnet50_epoch_14.pth \
        --perturbation-dir perturbed_constellations \
        --batch-size 256 \
        --seed 42 2>&1 | tee -a perturbation_evaluation_log.txt
    
    echo "================================================" | tee -a perturbation_evaluation_log.txt
    echo "Perturbation evaluation completed at $(date)" | tee -a perturbation_evaluation_log.txt
'

echo "✓ Perturbation evaluation started in screen session 'perturbation_eval'"
echo ""
echo "Commands:"
echo "  Monitor progress:    screen -r perturbation_eval"
echo "  Check log:          tail -f perturbation_evaluation_log.txt"
echo "  Detach from screen: Ctrl-A, then D"
echo "  List screens:       screen -ls"
echo ""
echo "The evaluation is now running in the background."
echo "Results will be saved to: papers/ELSP_Paper/results/perturbation_analysis/"