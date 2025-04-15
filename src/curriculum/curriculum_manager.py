import torch
from typing import List, Dict, Optional


class CurriculumManager:
    """
    Manages curriculum learning stages for SNR classification.
    
    Handles stage progression, accuracy tracking, and plateau detection
    based on validation SNR accuracy.
    
    References:
    [1] Bengio, Y., et al. (2009). "Curriculum Learning." ICML.
    [2] Pentina, A., et al. (2015). "Curriculum Learning of Multiple Tasks."
    [3] Zhang, Y., et al. (2020). "Curriculum Learning for Deep Learning-based Signal Processing."
    """
    
    def __init__(self, stages: List[Dict], patience: int, device: torch.device):
        """
        Initialize curriculum manager.
        
        Args:
            stages: List of dicts with 'snr_list' keys defining curriculum stages
            patience: Epochs without improvement before progression
            device: Device to place tensors on
        """
        self.stages = stages
        self.current_stage = 0
        self.patience = patience
        self.device = device
        self.best_snr_accuracy = torch.tensor(0.0, device=device)
        self.epochs_without_improvement = 0
        self.current_snr_list = stages[0]['snr_list']
        self.stage_history = []
        
        # Log initial stage
        self.stage_history.append({
            'stage': self.current_stage,
            'snr_list': self.current_snr_list,
            'epoch': 0
        })
        
        print(f"Curriculum initialized with {len(stages)} stages")
        print(f"Initial SNR list: {self.current_snr_list}")
        print(f"Patience: {patience} epochs")
        
    def should_progress(self, snr_accuracy) -> bool:
        """
        Determine if curriculum should progress to next stage.
        
        Args:
            snr_accuracy: Current SNR classification accuracy
            
        Returns:
            bool: True if curriculum should progress, False otherwise
        """
        print(f"\n{'*'*50}")
        print(f"CURRICULUM PROGRESS CHECK (Stage {self.current_stage}/{len(self.stages)-1}):")
        print(f"Current SNR list: {self.current_snr_list}")
        print(f"SNR accuracy: {snr_accuracy:.2f}%")
        print(f"Best accuracy so far: {self.best_snr_accuracy.item():.2f}%")
        print(f"Epochs without improvement: {self.epochs_without_improvement}")
        print(f"Patience threshold: {self.patience}")
        
        # Check if we're at the final stage
        if self.current_stage >= len(self.stages) - 1:
            print(f"Already at final stage ({self.current_stage}). Not progressing.")
            print(f"{'*'*50}\n")
            return False
        
        # Update best accuracy if current is better
        if snr_accuracy > self.best_snr_accuracy:
            old_best = self.best_snr_accuracy.item()
            self.best_snr_accuracy = torch.tensor(snr_accuracy, device=self.device)
            self.epochs_without_improvement = 0
            print(f"New best accuracy: {old_best:.2f}% → {snr_accuracy:.2f}%")
            print(f"Epochs without improvement reset to 0")
        else:
            self.epochs_without_improvement += 1
            print(f"No improvement in accuracy. Epochs without improvement: {self.epochs_without_improvement}")
        
        # Check if we should progress
        should_progress = self.epochs_without_improvement >= self.patience
        if should_progress:
            print(f"PROGRESSING TO NEXT STAGE: Stage {self.current_stage} → {self.current_stage + 1}")
            print(f"Current SNR list: {self.current_snr_list}")
            print(f"Next SNR list: {self.stages[self.current_stage + 1]['snr_list']}")
        else:
            print(f"Not progressing yet. Need {self.patience - self.epochs_without_improvement} more epochs without improvement.")
        
        print(f"{'*'*50}\n")
        return should_progress
        
    def get_next_stage(self) -> Optional[List[int]]:
        """
        Get next stage's SNR list if available.
        
        Returns:
            Optional[List[int]]: Next SNR list or None if at final stage
        """
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.best_snr_accuracy = torch.tensor(0.0, device=self.device)
            self.epochs_without_improvement = 0
            self.current_snr_list = self.stages[self.current_stage]['snr_list']
            
            # Log stage transition
            self.stage_history.append({
                'stage': self.current_stage,
                'snr_list': self.current_snr_list,
                'epoch': len(self.stage_history)
            })
            
            print(f"Advancing to curriculum stage {self.current_stage}")
            print(f"New SNR list: {self.current_snr_list}")
            print(f"Length of new SNR list: {len(self.current_snr_list)}")
            
            return self.current_snr_list
        
        print("Already at final curriculum stage")
        return None
    
    def get_current_stage_info(self) -> Dict:
        """
        Get information about current stage.
        
        Returns:
            Dict: Current stage information
        """
        return {
            'stage': self.current_stage,
            'snr_list': self.current_snr_list,
            'patience': self.patience,
            'best_snr_accuracy': self.best_snr_accuracy.item(),
            'epochs_without_improvement': self.epochs_without_improvement
        }
    
    def get_current_snr_list(self) -> List[int]:
        """
        Get current stage's SNR list.
        
        Returns:
            List[int]: Current SNR list
        """
        current_list = self.current_snr_list
        print(f"Current curriculum stage: {self.current_stage}")
        print(f"Current SNR list: {current_list}")
        print(f"Length of current SNR list: {len(current_list)}")
        return current_list 