import torch.nn as nn
import torch
import json


def test_model(model, X_test, Y_test):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        # 1. Forward Pass
        output, _ = model(X_test)
        
        # 2. Reshape for Loss
        flat_output = output.view(-1, output.size(-1))
        flat_expected = Y_test.view(-1)
        
        # 3. Calculate Loss
        loss = criterion(flat_output, flat_expected)
        
        # 4. Calculate Accuracy
        scores, indices = torch.max(output.data, 2)
        correct_guesses = (indices == Y_test)
        accuracy = 100 * (correct_guesses.sum().item() / Y_test.numel())
        
        print(f"Test Loss: {loss.item():.4f} | Test Accuracy: {accuracy:.2f}%")
        
        return accuracy