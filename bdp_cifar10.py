import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os

# --- Global Settings and Optimized Hyperparameters ---
MODEL_PATH = './resnet18_cifar10.pth'
NUM_CLASSES = 10
# Optimized for higher accuracy
EPOCHS = 200 
BATCH_SIZE = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Used Device: {DEVICE}")

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed()

# --- Boundary Differentially Private Layer (BDPL) Implementation ---
def boundary_differentially_private_layer(logits: torch.Tensor, epsilon: float, delta: float) -> torch.Tensor:
    """
    Implements Boundary Differentially Private Layer (BDPL) - Pairwise Boundary Randomized Response.
    """
    
    # Ensure epsilon is a Tensor on the correct device for torch operations
    epsilon_tensor = torch.tensor(epsilon, dtype=torch.float32).to(logits.device)
    
    # 1. Calculate the flip probability p_flip
    p_flip = torch.exp(-epsilon_tensor) / (1 + torch.exp(-epsilon_tensor))
    
    # 2. Get top two Logits
    top_2_values, top_2_indices = torch.topk(logits, 2, dim=1)
    
    winner_score = top_2_values[:, 0]
    counter_score = top_2_values[:, 1]
    
    # 3. Determine the sensitive boundary region Z: score difference < Delta
    score_difference = winner_score - counter_score
    is_sensitive = (score_difference < delta)
    
    perturbed_logits = logits.clone()
    sensitive_indices = torch.where(is_sensitive)[0]
    
    if len(sensitive_indices) > 0:
        random_draws = torch.rand(len(sensitive_indices), device=logits.device)
        p_flip_value = p_flip.item() 
        
        for i_local, i_global in enumerate(sensitive_indices):
            if random_draws[i_local] < p_flip_value:
                winner_idx = top_2_indices[i_global, 0]
                counter_idx = top_2_indices[i_global, 1]
                
                temp_logit = perturbed_logits[i_global, winner_idx].clone()
                perturbed_logits[i_global, winner_idx] = perturbed_logits[i_global, counter_idx].clone()
                perturbed_logits[i_global, counter_idx] = temp_logit.clone()
            
    return perturbed_logits

# --- Model and Data Handling Functions ---

def setup_data():
    """Download and load CIFAR-10, with Data Augmentation for training."""
    print("\n--- Step 1: Downloading CIFAR-10 Dataset ---")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return trainloader, testloader

def setup_model(load_weights=False):
    """Load ResNet-18 model."""
    model = torchvision.models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)
    
    if load_weights and os.path.exists(MODEL_PATH):
        print(f"Loading trained weights from: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model
    
    return model

def train_model(model, trainloader, testloader, epochs=EPOCHS):
    """Train ResNet-18 model and save the best checkpoint."""
    print(f"\n--- Step 2: Training ResNet-18 Model ({epochs} Epochs) ---")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        
        for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} (Train)")):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        acc = evaluate_utility(model, testloader, is_private=False)
        print(f"Epoch {epoch+1} Test Accuracy: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)
            
    print(f"\n--- Step 3: Training complete. (Best Accuracy: {best_acc:.2f}%) ---")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def evaluate_utility(model, testloader, is_private=False, epsilon=1.0, delta=0.5):
    """Evaluate model accuracy (with optional BDP)."""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            raw_logits = model(inputs)
            
            if is_private:
                logits = boundary_differentially_private_layer(raw_logits, epsilon=epsilon, delta=delta)
            else:
                logits = raw_logits
            
            _, predicted = torch.max(logits.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# --- BDP Utility Evaluation and Plotting (Optimized Font/Size) ---
def run_bdp_evaluation(model, testloader):
    """Execute BDP test and plot the Privacy-Utility Trade-off with larger fonts and size."""
    print("\n--- Step 4: BDP Privacy-Utility Trade-off Evaluation ---")
    
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
    DELTA_THRESHOLD = 0.5 
    print(f"Using Boundary Sensitivity Threshold Δ = {DELTA_THRESHOLD}")

    accuracy_results = []
    
    for eps in tqdm(epsilons, desc="BDP Utility Test"):
        epsilon_val = eps if eps != float('inf') else 10000.0 
        
        acc = evaluate_utility(model, testloader, 
                                is_private=True, 
                                epsilon=epsilon_val, 
                                delta=DELTA_THRESHOLD)
        accuracy_results.append(acc)
    
    baseline_accuracy = accuracy_results[-1]
    
    print("\n--- Test Results ---")
    for eps, acc in zip(epsilons, accuracy_results):
        print(f"ε = {eps:<6}: Accuracy = {acc:.2f}%")
    print(f"Baseline (Non-Private) Accuracy: {baseline_accuracy:.2f}%")
    
    # --- Plotting Comparison Chart (Optimized Size and Fonts) ---
    x_labels = [str(e) for e in epsilons[:-1]] + ['Non-Private']
    
    # 放大圖表尺寸
    plt.figure(figsize=(14, 8)) 
    
    # 繪製線圖 (圖例字體大小設定)
    plt.plot(x_labels, accuracy_results, marker='o', linestyle='-', color='blue', 
             label='BDP Accuracy')
    plt.axhline(y=baseline_accuracy, color='r', linestyle='--', 
                label=f'Baseline Acc ({baseline_accuracy:.2f}%)')
    
    # 設定圖表標題字體大小
    plt.title(f'BDP Privacy-Utility Trade-off (ResNet-18 on CIFAR-10, $\\Delta$={DELTA_THRESHOLD})', 
              fontsize=16)
    
    # 設定軸標籤字體大小
    plt.xlabel('Privacy Budget $\\epsilon$', fontsize=14)
    plt.ylabel('Test Accuracy (%)', fontsize=14)
    
    # 設定圖例字體大小
    plt.legend(fontsize=12) 
    
    # 設定刻度標籤字體大小 (X軸和Y軸)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.grid(True, linestyle='--')
    plt.show()

# --- Main Execution Block ---
if __name__ == '__main__':
    trainloader, testloader = setup_data()

    # --- Phase 1: Train or Load Model ---
    model = setup_model(load_weights=True)
    
    if model is None or not os.path.exists(MODEL_PATH):
        print("\nStarting model training.")
        model = setup_model(load_weights=False)
        model = train_model(model, trainloader, testloader)
    else:
        initial_acc = evaluate_utility(model, testloader, is_private=False)
        print(f"Loaded Model Initial Accuracy (Non-Private): {initial_acc:.2f}%")
        
    # --- Phase 2: BDP Test ---
    if model:
        run_bdp_evaluation(model, testloader)