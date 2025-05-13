"""
Unit tests for the 4DHands module.
"""

import os
import sys
import unittest
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hands_4d import RAT, SIR, Hands4D


class TestRAT(unittest.TestCase):
    """Tests for Relation-aware Two-Hand Tokenization module."""
    
    def setUp(self):
        """Set up the test environment."""
        self.config = {
            'input_dim': 3,
            'hidden_dim': 64,
            'num_fingers': 5,
            'joints_per_finger': 4,
            'num_heads': 2
        }
        self.rat = RAT(self.config)
        
    def test_rat_forward(self):
        """Test the forward pass of the RAT module."""
        # Create a batch of hand data [batch_size, seq_len, num_hands, num_keypoints, input_dim]
        batch_size = 2
        seq_len = 5
        num_hands = 2
        num_keypoints = self.config['num_fingers'] * self.config['joints_per_finger'] + 1  # +1 for wrist
        
        # Create random input data
        hand_data = torch.randn(batch_size, seq_len, num_hands, num_keypoints, self.config['input_dim'])
        
        # Forward pass
        output = self.rat(hand_data)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, num_hands, self.config['hidden_dim'])
        self.assertEqual(output.shape, expected_shape)
        
    def test_rat_small_input(self):
        """Test RAT with minimal input size."""
        hand_data = torch.randn(1, 1, 1, 21, 3)
        output = self.rat(hand_data)
        self.assertEqual(output.shape, (1, 1, 1, self.config['hidden_dim']))


class TestSIR(unittest.TestCase):
    """Tests for Spatio-temporal Interaction Reasoning module."""
    
    def setUp(self):
        """Set up the test environment."""
        self.config = {
            'hidden_dim': 64,
            'num_heads': 2,
            'num_layers': 2,
            'dropout': 0.1,
            'num_classes': 10
        }
        self.sir = SIR(self.config)
        
    def test_sir_forward(self):
        """Test the forward pass of the SIR module."""
        # Create hand tokens [batch_size, seq_len, num_hands, hidden_dim]
        batch_size = 2
        seq_len = 5
        num_hands = 2
        
        # Create random input data
        hand_tokens = torch.randn(batch_size, seq_len, num_hands, self.config['hidden_dim'])
        
        # Forward pass
        hand_features, logits = self.sir(hand_tokens)
        
        # Check output shapes
        self.assertEqual(hand_features.shape, (batch_size, seq_len, self.config['hidden_dim']))
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config['num_classes']))
        
    def test_sir_without_classification(self):
        """Test SIR without classification head."""
        # Create SIR without classification
        config_no_classes = self.config.copy()
        config_no_classes['num_classes'] = None
        sir_no_classes = SIR(config_no_classes)
        
        # Create input data
        hand_tokens = torch.randn(2, 5, 2, self.config['hidden_dim'])
        
        # Forward pass
        hand_features, logits = sir_no_classes(hand_tokens)
        
        # Check output shapes
        self.assertEqual(hand_features.shape, (2, 5, self.config['hidden_dim']))
        self.assertIsNone(logits)


class TestHands4D(unittest.TestCase):
    """Tests for the complete Hands4D module."""
    
    def setUp(self):
        """Set up the test environment."""
        self.config = {
            'input_dim': 3,
            'hidden_dim': 64,
            'num_fingers': 5,
            'joints_per_finger': 4,
            'num_heads': 2,
            'num_layers': 2,
            'dropout': 0.1,
            'num_classes': 10
        }
        self.hands4d = Hands4D(self.config)
        
    def test_hands4d_forward(self):
        """Test the forward pass of the Hands4D module."""
        # Create a batch of hand data [batch_size, seq_len, num_hands, num_keypoints, input_dim]
        batch_size = 2
        seq_len = 5
        num_hands = 2
        num_keypoints = self.config['num_fingers'] * self.config['joints_per_finger'] + 1  # +1 for wrist
        
        # Create random input data
        hand_data = torch.randn(batch_size, seq_len, num_hands, num_keypoints, self.config['input_dim'])
        
        # Forward pass
        hand_features, logits = self.hands4d(hand_data)
        
        # Check output shapes
        self.assertEqual(hand_features.shape, (batch_size, seq_len, self.config['hidden_dim']))
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config['num_classes']))
        
    def test_gradients(self):
        """Test gradient flow in the Hands4D module."""
        # Create input data that requires gradient
        hand_data = torch.randn(1, 3, 2, 21, 3, requires_grad=True)
        
        # Forward pass
        hand_features, logits = self.hands4d(hand_data)
        
        # Create dummy target for classification
        target = torch.zeros(1, 3, dtype=torch.long)
        
        # Compute loss and backpropagate
        loss = torch.nn.functional.cross_entropy(logits.view(-1, self.config['num_classes']), target.view(-1))
        loss.backward()
        
        # Check if gradients are computed
        self.assertIsNotNone(hand_data.grad)
        self.assertTrue(torch.any(hand_data.grad != 0))


if __name__ == '__main__':
    unittest.main()
