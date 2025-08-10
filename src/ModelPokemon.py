import torch.nn as nn

class ModelPokemon(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(24944, 512, device="cuda"),
            nn.LayerNorm(512, device="cuda"),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256, device="cuda"), 
            nn.ReLU(),   
            nn.Linear(256, 128, device="cuda"), 
            nn.ReLU(),   
            nn.Linear(128, 64, device="cuda"), 
            nn.ReLU(),   
            nn.Linear(64, 32, device="cuda"), 
            nn.ReLU(),  
            nn.Linear(32, 16, device="cuda"), 
            nn.ReLU(),
            nn.Linear(16, 8, device="cuda"),
        )
        
    def forward(self, x):
        return self.model(x)
