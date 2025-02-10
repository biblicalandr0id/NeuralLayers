import torch
import numpy as np

class NLNDExample:
    def __init__(self):
        self.creation_utc = "2025-02-10 13:23:54"
        self.creator = "biblicalandr0id"
        
        # Spatial dimensions
        self.dx = self.dy = self.dz = 32
        
        # Initialize example voltage fields
        self.gaussian_field = self.create_gaussian_field()
        self.oscillating_field = self.create_oscillating_field()
        self.wave_packet_field = self.create_wave_packet()
        self.logical_pattern_field = self.create_logical_pattern()
        self.combined_field = self.create_combined_field()

    def create_gaussian_field(self):
        x = y = z = torch.linspace(-2, 2, 32)
        X, Y, Z = torch.meshgrid(x, y, z)
        return torch.exp(-(X**2 + Y**2 + Z**2))

    def create_oscillating_field(self):
        t = torch.linspace(0, 2*np.pi, 32)
        return torch.sin(t).reshape(1, -1, 1).expand(32, 32, 32)

    def create_wave_packet(self):
        k = 2*np.pi
        x = torch.linspace(-5, 5, 32)
        X, Y, Z = torch.meshgrid(x, x, x)
        return torch.exp(-(X**2 + Y**2 + Z**2)) * torch.cos(k*X)

    def create_logical_pattern(self):
        pattern = torch.zeros(32, 32, 32)
        pattern[::2, ::2, ::2] = 1  # Create logical truth value pattern
        return pattern

    def create_combined_field(self):
        return (self.gaussian_field + 
                self.oscillating_field + 
                self.wave_packet_field + 
                self.logical_pattern_field) / 4

    def get_time_points(self, duration=1.0, steps=100):
        return torch.linspace(0, duration, steps)

    def generate_initial_condition(self, field_type='combined'):
        if field_type == 'gaussian':
            return self.gaussian_field
        elif field_type == 'oscillating':
            return self.oscillating_field
        elif field_type == 'wave_packet':
            return self.wave_packet_field
        elif field_type == 'logical':
            return self.logical_pattern_field
        else:
            return self.combined_field

if __name__ == "__main__":
    nlnd_examples = NLNDExample()
    
    # Get example initial condition
    V_initial = nlnd_examples.generate_initial_condition()
    
    # Get time points for evolution
    t_span = nlnd_examples.get_time_points()
    
    print(f"Creation Time (UTC): {nlnd_examples.creation_utc}")
    print(f"Creator: {nlnd_examples.creator}")
    print(f"\nInitial Field Shape: {V_initial.shape}")
    print(f"Time Points: {len(t_span)}")
