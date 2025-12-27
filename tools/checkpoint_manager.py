#!/usr/bin/env python3
"""
Checkpoint management utilities for TinyLLM.
Provides functions to list, load, and manage model checkpoints.
"""

import os
import torch
from typing import List, Dict, Optional, Any
from config import ModelConfig
from model.transformer import TinyLLM

class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints"):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir (str): Directory containing checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints in the directory.
        
        Returns:
            List[Dict]: List of checkpoint info dictionaries
        """
        checkpoints = []
        
        if not os.path.exists(self.checkpoint_dir):
            return checkpoints
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.pt'):
                filepath = os.path.join(self.checkpoint_dir, filename)
                try:
                    info = self.get_checkpoint_info(filepath)
                    if info:
                        info['filename'] = filename
                        info['path'] = filepath
                        checkpoints.append(info)
                except Exception as e:
                    print(f"Error reading checkpoint {filename}: {e}")
        
        # Sort by step (most recent first)
        checkpoints.sort(key=lambda x: x.get('step', 0), reverse=True)
        return checkpoints
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
        
        Returns:
            Dict or None: Checkpoint information
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            info = {
                'step': checkpoint.get('step', 0),
                'epoch': checkpoint.get('epoch', 0),
                'model_keys': list(checkpoint.get('model', {}).keys()),
                'optimizer_keys': list(checkpoint.get('optimizer', {}).keys()),
                'file_size': os.path.getsize(checkpoint_path),
                'file_path': checkpoint_path
            }
            
            return info
        except Exception as e:
            print(f"Error reading checkpoint info: {e}")
            return None
    
    def load_model_from_checkpoint(self, checkpoint_path: str, device="auto") -> TinyLLM:
        """
        Load a model from a checkpoint file.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
            device (str): Device to load the model on
        
        Returns:
            TinyLLM: Loaded model
        """
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        
        # Create model
        model = TinyLLM(ModelConfig()).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        return model
    
    def print_checkpoint_summary(self, checkpoint_path: str):
        """
        Print a summary of checkpoint information.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        info = self.get_checkpoint_info(checkpoint_path)
        if not info:
            print(f"No checkpoint info available for {checkpoint_path}")
            return
        
        print(f"\n=== Checkpoint Summary ===")
        print(f"File: {os.path.basename(checkpoint_path)}")
        print(f"Path: {checkpoint_path}")
        print(f"Step: {info.get('step', 'N/A')}")
        print(f"Epoch: {info.get('epoch', 'N/A')}")
        print(f"File Size: {info.get('file_size', 0) / 1024 / 1024:.2f} MB")
        print(f"Model Keys: {len(info.get('model_keys', []))}")
        print(f"Optimizer Keys: {len(info.get('optimizer_keys', []))}")
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the path to the most recent checkpoint.
        
        Returns:
            str or None: Path to the latest checkpoint
        """
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[0]['path']
        return None
    
    def compare_checkpoints(self, checkpoint1_path: str, checkpoint2_path: str):
        """
        Compare two checkpoints and show differences.
        
        Args:
            checkpoint1_path (str): Path to first checkpoint
            checkpoint2_path (str): Path to second checkpoint
        """
        info1 = self.get_checkpoint_info(checkpoint1_path)
        info2 = self.get_checkpoint_info(checkpoint2_path)
        
        if not info1 or not info2:
            print("Could not compare checkpoints - info unavailable")
            return
        
        print(f"\n=== Checkpoint Comparison ===")
        print(f"Checkpoint 1: {os.path.basename(checkpoint1_path)}")
        print(f"  Step: {info1.get('step', 'N/A')}")
        print(f"  Epoch: {info1.get('epoch', 'N/A')}")
        print(f"  Size: {info1.get('file_size', 0) / 1024 / 1024:.2f} MB")
        
        print(f"\nCheckpoint 2: {os.path.basename(checkpoint2_path)}")
        print(f"  Step: {info2.get('step', 'N/A')}")
        print(f"  Epoch: {info2.get('epoch', 'N/A')}")
        print(f"  Size: {info2.get('file_size', 0) / 1024 / 1024:.2f} MB")
        
        # Compare model keys
        keys1 = set(info1.get('model_keys', []))
        keys2 = set(info2.get('model_keys', []))
        
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        
        if only_in_1:
            print(f"\nKeys only in checkpoint 1: {only_in_1}")
        if only_in_2:
            print(f"\nKeys only in checkpoint 2: {only_in_2}")
        if not only_in_1 and not only_in_2:
            print(f"\nModel architectures are identical")

def main():
    """Example usage of the checkpoint manager."""
    manager = CheckpointManager()
    
    print("=== TinyLLM Checkpoint Manager ===")
    
    # List all checkpoints
    print("\nAvailable checkpoints:")
    checkpoints = manager.list_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found.")
        return
    
    for i, ckpt in enumerate(checkpoints):
        print(f"{i+1}. {ckpt['filename']} (step: {ckpt.get('step', 'N/A')}, epoch: {ckpt.get('epoch', 'N/A')})")
    
    # Show info for the latest checkpoint
    latest = manager.get_latest_checkpoint()
    if latest:
        print(f"\nLatest checkpoint: {os.path.basename(latest)}")
        manager.print_checkpoint_summary(latest)
    
    # Example: Load and use the latest checkpoint
    if latest:
        print(f"\nLoading model from {os.path.basename(latest)}...")
        model = manager.load_model_from_checkpoint(latest)
        print(f"Model loaded successfully on {next(model.parameters()).device}")

if __name__ == "__main__":
    main()