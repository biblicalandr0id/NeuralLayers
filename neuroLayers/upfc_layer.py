import torch
import torch.nn as nn
import numpy as np

class UnifiedProccessingFunctionCerebrum(nn.Module):
    def __init__(self, num_regions=6, cortical_dims=(64,64,64)):
        super().__init__()
        self.creation_utc = "2025-02-10 15:14:56"
        self.creator = "biblicalandr0id"
        
        self.R = num_regions
        self.V1_to_V6 = cortical_dims
        
        self.phi = nn.ParameterList([
            nn.Parameter(torch.randn(cortical_dims)) 
            for _ in range(num_regions)
        ])
        
        self.W_cortical = nn.Parameter(torch.randn(cortical_dims))
        self.sigma = nn.Sigmoid()

    def laplacian_phi(self, phi_r):
        dx = torch.diff(phi_r, dim=0, prepend=phi_r[:1], append=phi_r[-1:])
        dy = torch.diff(phi_r, dim=1, prepend=phi_r[:,:1], append=phi_r[:,-1:])
        dz = torch.diff(phi_r, dim=2, prepend=phi_r[:,:,:1], append=phi_r[:,:,-1:])
        return dx + dy + dz

    def volume_integral(self, field):
        return torch.trapz(
            torch.trapz(
                torch.trapz(field, dx=1.0),
            dx=1.0),
        dx=1.0)

    def compose_psi_gamma(self, psi, gamma):
        return psi * gamma

    def forward(self, psi, gamma):
        regional_sum = 0
        
        for r in range(self.R):
            # Compute ∇²Φᵣ
            del2_phi = self.laplacian_phi(self.phi[r])
            
            # Multiply by cortical weights
            weighted_field = del2_phi * self.W_cortical
            
            # Apply activation
            activated_field = self.sigma(weighted_field)
            
            # Volume integral over V₁→V₆
            integrated = self.volume_integral(activated_field)
            
            regional_sum += integrated

        # Tensor product with composition (Ψ ∘ Γ)
        psi_gamma = self.compose_psi_gamma(psi, gamma)
        
        cerebrum_output = regional_sum * psi_gamma

        return {
            'UPF_cerebrum': cerebrum_output,
            'regional_activations': [self.phi[r].data for r in range(self.R)],
            'cortical_weights': self.W_cortical.data,
            'metadata': {
                'creation_utc': self.creation_utc,
                'creator': self.creator,
                'num_regions': self.R,
                'cortical_dims': self.V1_to_V6
            }
        }