import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BrainNetwork(nn.Module):
    def __init__(self):
        super(BrainNetwork, self).__init__()
        
        # Constants (Actual biophysical values)
        self.Vrest = -70.0  # mV (Resting potential)
        self.Vpeak = 40.0   # mV (Peak potential)
        self.tau = 20.0     # ms (Time constant)
        self.theta = -55.0  # mV (Firing threshold)
        self.eta = 0.001    # Learning rate
        self.rho = 0.1      # Plasticity coefficient
        self.dt = 0.1       # ms (Time step)
        
        # Calcium dynamics constants
        self.Ca_0 = 100.0   # nM (Baseline calcium)
        self.Ca_max = 1000.0  # nM (Maximum calcium)
        self.Ca_tau = 50.0    # ms (Calcium decay constant)
        
        # Energy parameters
        self.ATP_min = 1000.0  # μM (Minimum ATP required)
        self.ATP_0 = 5000.0    # μM (Initial ATP)
        self.ATP_max = 10000.0 # μM (Maximum ATP)
        
        # Neurotransmitter parameters
        self.NT_0 = 1.0      # μM (Baseline NT)
        self.NT_max = 10.0   # μM (Maximum NT)
        self.NT_tau = 100.0  # ms (NT decay constant)
        
        # Initialize weight matrices with biological constraints
        self.initialize_weights()
        
        # Neural layers
        self.sensory_layer = nn.Linear(6, 128)  # 6 sensory inputs
        self.primary_layer = nn.Linear(128, 256)
        self.association_layer = nn.Linear(256, 512)
        self.higher_layer = nn.Linear(512, 1024)
        
        # Parallel processing layers
        self.limbic_layer = nn.Linear(1024, 256)
        self.basal_layer = nn.Linear(1024, 256)
        self.cerebellum_layer = nn.Linear(1024, 256)
        
        # Sequential processing layers
        self.memory_layer = nn.Linear(1024, 512)
        self.executive_layer = nn.Linear(512, 256)
        
        # Output layers
        self.motor_output = nn.Linear(256, 64)
        self.autonomic_output = nn.Linear(256, 32)
        self.cognitive_output = nn.Linear(256, 128)
        
    def initialize_weights(self):
        # Initialize sensory weights (Wi)
        self.W_visual = torch.randn(64, 64) * 0.1
        self.W_auditory = torch.randn(32, 32) * 0.1
        self.W_tactile = torch.randn(32, 32) * 0.1
        self.W_olfactory = torch.randn(16, 16) * 0.1
        self.W_gustatory = torch.randn(16, 16) * 0.1
        self.W_proprioceptive = torch.randn(32, 32) * 0.1
        
        # Clamp weights to [0, 1]
        self.W_visual.data.clamp_(0, 1)
        self.W_auditory.data.clamp_(0, 1)
        self.W_tactile.data.clamp_(0, 1)
        self.W_olfactory.data.clamp_(0, 1)
        self.W_gustatory.data.clamp_(0, 1)
        self.W_proprioceptive.data.clamp_(0, 1)
        
    def membrane_potential(self, V, input_current):
        """Calculate membrane potential evolution"""
        dV = (-(V - self.Vrest) + input_current) * (self.dt / self.tau)
        V_new = V + dV
        return torch.clamp(V_new, self.Vrest, self.Vpeak)
    
    def calcium_dynamics(self, V, Ca):
        """Simulate calcium dynamics"""
        influx = torch.sigmoid((V - self.theta) / 10) * (self.Ca_max - Ca)
        efflux = Ca * (self.dt / self.Ca_tau)
        dCa = influx - efflux
        return torch.clamp(Ca + dCa, 0, self.Ca_max)
    
    def energy_dynamics(self, ATP, activity):
        """Simulate ATP consumption and production"""
        consumption = activity.abs().mean() * self.dt
        production = (self.ATP_max - ATP) * (self.dt / 1000)
        dATP = production - consumption
        return torch.clamp(ATP + dATP, self.ATP_min, self.ATP_max)
    
    def neurotransmitter_dynamics(self, NT, V):
        """Simulate neurotransmitter kinetics"""
        release = torch.sigmoid((V - self.theta) / 10) * (self.NT_max - NT)
        degradation = NT * (self.dt / self.NT_tau)
        dNT = release - degradation
        return torch.clamp(NT + dNT, 0, self.NT_max)
    
    def forward(self, sensory_input, state=None):
        # Initialize or unpack state
        if state is None:
            V = torch.full_like(sensory_input[:, 0:1], self.Vrest)
            Ca = torch.full_like(sensory_input[:, 0:1], self.Ca_0)
            ATP = torch.full_like(sensory_input[:, 0:1], self.ATP_0)
            NT = torch.full_like(sensory_input[:, 0:1], self.NT_0)
        else:
            V, Ca, ATP, NT = state
        
        # Energy scaling factor
        energy_scale = ATP / self.ATP_min
        
        # 1. Sensory Processing
        sensory = self.sensory_layer(sensory_input * energy_scale)
        V = self.membrane_potential(V, sensory.mean(dim=1, keepdim=True))
        
        # 2. Primary Processing
        primary = F.relu(self.primary_layer(sensory)) * (1 - torch.exp(-V/self.tau))
        Ca = self.calcium_dynamics(V, Ca)
        
        # 3. Association Processing
        association = torch.tanh(self.association_layer(primary) / self.theta)
        NT = self.neurotransmitter_dynamics(NT, V)
        
        # 4. Higher Processing
        higher = torch.sigmoid(self.higher_layer(association)) * (1 + self.rho)
        ATP = self.energy_dynamics(ATP, higher)
        
        # 5. Parallel Processing
        limbic = torch.sigmoid(self.limbic_layer(higher)) * (Ca / self.Ca_0)
        basal = F.relu(self.basal_layer(higher)) * energy_scale
        cerebellum = torch.tanh(self.cerebellum_layer(higher)) * (NT / self.NT_0)
        
        # 6. Sequential Processing
        memory = self.memory_layer(higher) * torch.exp(torch.tensor(-self.dt/self.tau))
        executive = self.executive_layer(memory) * (1 - torch.exp(-ATP/self.ATP_min))
        
        # 7. Output Generation
        motor = self.motor_output(executive) * energy_scale
        autonomic = torch.sigmoid(self.autonomic_output(executive))
        cognitive = self.cognitive_output(executive) * (NT / self.NT_0)
        
        # Package outputs and state
        outputs = {
            'motor': motor,
            'autonomic': autonomic,
            'cognitive': cognitive,
            'membrane_potential': V,
            'calcium': Ca,
            'ATP': ATP,
            'neurotransmitter': NT
        }
        
        state = (V, Ca, ATP, NT)
        
        return outputs, state

# Example usage
if __name__ == "__main__":
    # Initialize network
    brain = BrainNetwork()
    
    # Create sample input (batch_size=1, 6 sensory inputs)
    sensory_input = torch.rand(1, 6)
    
    # Forward pass
    outputs, state = brain(sensory_input)
    
    print("Network State:")
    print(f"Membrane Potential: {outputs['membrane_potential'].item():.2f} mV")
    print(f"Calcium Level: {outputs['calcium'].item():.2f} nM")
    print(f"ATP Level: {outputs['ATP'].item():.2f} μM")
    print(f"Neurotransmitter Level: {outputs['neurotransmitter'].item():.2f} μM")
    print("\nOutputs:")
    print(f"Motor Output Shape: {outputs['motor'].shape}")
    print(f"Autonomic Output Shape: {outputs['autonomic'].shape}")
    print(f"Cognitive Output Shape: {outputs['cognitive'].shape}")