import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from src.model import ToyModel



def train_pe_variants(model,X,X_val,Y,Y_val):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Loss Function
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(),lr=0.001)

    epochs=10




    history=[]




    total_steps = 0

    for epoch in range(epochs):
        model.train()
    
        
        total_time = 0
        running_loss = 0
        correct = 0
        
        average_training_time = 0
        
        total_elements = 0
        # --- 1. RESET MEMORY HERE ---
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            
        start_time = time.time()
        
        
        optimizer.zero_grad()
        
        # Forward Pass
        output,weights = model(X)
        
        # print(f"This is the shape of Y: {Y.shape}")
        # print(f"This is the shape of ouput: {output.shape}")

        # Flatten predictions to [Batch * Seq_Len, Vocab_Size] -> [10, 512]
        flat_output = output.view(-1, output.size(-1))
        
        # Flatten labels to [Batch * Seq_Len] -> [160]
        flat_expected_labels = Y.view(-1)
        loss = criterion(flat_output,flat_expected_labels)
        
        #Backward pass
        loss.backward()
            
            
        optimizer.step()
            
            
        #stop timer safely
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
            
        total_time += (end_time - start_time)
            
        running_loss += loss.item()
        
    
        scores,indices = torch.max(output.data,2)
        
        # print(f"This is the shape of scores {scores.shape}")
        # print(f"This is the shape of indices {indices.shape}")

    
        # print(f"This is the predicted: {predicted}")
        
        correct_guesses = (indices == Y)
            
        total_steps += 1
        #is_close = torch.abs(output - Y) <= tolerance
        correct += correct_guesses.sum().item()
            
        total_elements = Y.numel()
        
        
        #Printing stats per epoch
        train_loss = running_loss/total_steps
        
        average_training_time = total_time /total_steps
        
        training_accuracy =  100 * correct /total_elements
        
        peak_memory_mb = 0
        if device.type == "cuda":
            peak_memory_bytes = torch.cuda.max_memory_allocated(device)
            peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        
        
        model.eval()
        
        val_loss = 0
        total_time = 0
        total_elements = 0
        is_close = 0
        total_steps = 0
        running_loss =0
        
    
        
        
        with torch.no_grad():
        
                #start timer safely
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.time()
                
                
                # Forward Pass
                output,_ = model(X_val)
                
                # Flatten predictions to [Batch * Seq_Len, Vocab_Size] -> [10, 512]
                flat_output_val= output.view(-1, output.size(-1))
                
                # Flatten labels to [Batch * Seq_Len] -> [160]
                flat_expected_labels_val = Y_val.view(-1)
                
                loss = criterion(flat_output_val,flat_expected_labels_val)
                                
                
                #stop timer safely
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()
                
                total_time += (end_time - start_time)
                
                val_loss += loss.item()
                
                scores,indices = torch.max(output.data,2)
        
                # print(f"This is the shape of scores {scores.shape}")
                # print(f"This is the shape of indices {indices.shape}")

            
                # print(f"This is the predicted: {predicted}")
                
                correct_guesses = (indices == Y_val)
                    
                total_steps += 1
                #is_close = torch.abs(output - Y) <= tolerance
                correct += correct_guesses.sum().item()
                
                # is_close = torch.abs(output - Y_val) <= tolerance
                # correct += is_close.sum().item()
                
                total_elements = Y_val.numel()
                
            
                    
                    
                
                
        val_loss = val_loss/total_steps
        
        average_validation_time = total_time /total_steps
        
        validation_accuracy =  100 * correct /total_elements

        
        metrics = {
        
            "Epoch":epoch + 1,
            "train_loss": train_loss,
            "training_acc": training_accuracy,
            "training_time":average_training_time,
            "val_loss":val_loss,
            "val_time": average_validation_time,
            "val_accuracy": validation_accuracy,
            "training_memory_mb":peak_memory_mb
        }
        
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Accuracy: {training_accuracy:.2f}%")
        print(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f} | Val Accuracy: {validation_accuracy:.2f}%")

        
        history.append(metrics)
        
    with open(f"./../experiments/complexity_comparion.json","w") as f:
        json.dump(history,f,indent=4)
        
        
    return history
        
        