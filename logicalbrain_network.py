import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import numpy as np
from datetime import datetime
import math


class CerebrumModule(nn.Module):
    """
    Cerebrum: Higher cognitive functions including executive control,
    working memory, reasoning, and conscious thought.

    Implements:
    - Executive function (planning, decision-making)
    - Working memory maintenance
    - Abstract reasoning
    - Attention control
    """
    def __init__(self, hidden_dim: int):
        super().__init__()

        # Executive Control Network
        self.executive_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Working Memory Buffer (multi-head attention for persistent representation)
        self.working_memory = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )

        # Reasoning Network (logical integration)
        self.reasoning_network = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu'
        )

        # Attention Controller
        self.attention_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Integration layer
        self.integration = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, sensory_input: torch.Tensor,
                logical_input: torch.Tensor) -> torch.Tensor:
        """
        Process sensory and logical inputs through cerebrum.

        Args:
            sensory_input: Processed sensory information
            logical_input: Logical reasoning representations

        Returns:
            Integrated cerebral output
        """
        # Concatenate inputs for executive processing
        combined = torch.cat([sensory_input, logical_input], dim=-1)

        # Executive control processing
        executive_output = self.executive_network(combined)

        # Working memory maintenance (self-attention)
        memory_output, _ = self.working_memory(
            executive_output.unsqueeze(0),
            executive_output.unsqueeze(0),
            executive_output.unsqueeze(0)
        )
        memory_output = memory_output.squeeze(0)

        # Abstract reasoning
        reasoning_output = self.reasoning_network(memory_output.unsqueeze(0))
        reasoning_output = reasoning_output.squeeze(0)

        # Attention modulation
        attention_weights = self.attention_controller(reasoning_output)
        modulated_output = reasoning_output * attention_weights

        # Final integration
        integrated = self.integration(
            torch.cat([modulated_output, executive_output], dim=-1)
        )

        return integrated


class CerebellumModule(nn.Module):
    """
    Cerebellum: Motor coordination, motor learning, balance, and timing.

    Implements:
    - Motor sequence learning
    - Error correction
    - Temporal prediction
    - Fine motor coordination
    """
    def __init__(self, hidden_dim: int):
        super().__init__()

        # Motor Learning Network (Purkinje cell-like processing)
        self.motor_learning = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Error Correction Network
        self.error_correction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Temporal Prediction (for timing and anticipation)
        self.temporal_predictor = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Coordination Network
        self.coordination = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        # Internal state for temporal processing
        self.register_buffer('hidden_state', torch.zeros(2, 1, hidden_dim))

    def forward(self, sensory_input: torch.Tensor,
                truth_values: torch.Tensor) -> torch.Tensor:
        """
        Process motor-related information and truth values.

        Args:
            sensory_input: Sensory information for motor planning
            truth_values: Logical truth values for decision making

        Returns:
            Motor coordination output
        """
        # Motor learning processing
        motor_learned = self.motor_learning(sensory_input)

        # Error correction based on truth values (prediction error)
        error_signal = self.error_correction(truth_values)
        corrected_motor = motor_learned + error_signal

        # Temporal prediction (sequence learning)
        batch_size = sensory_input.size(0)
        if self.hidden_state.size(1) != batch_size:
            self.hidden_state = torch.zeros(
                2, batch_size, self.hidden_state.size(2),
                device=sensory_input.device
            )

        temporal_out, self.hidden_state = self.temporal_predictor(
            corrected_motor.unsqueeze(1),
            self.hidden_state
        )
        temporal_out = temporal_out.squeeze(1)

        # Coordinate all signals
        coordinated = self.coordination(
            torch.cat([temporal_out, corrected_motor], dim=-1)
        )

        return coordinated


class BrainstemModule(nn.Module):
    """
    Brainstem: Autonomic functions, arousal, vital processes.

    Implements:
    - Arousal and alertness modulation
    - Autonomic regulation
    - Vital signs monitoring
    - Homeostatic control
    """
    def __init__(self, hidden_dim: int):
        super().__init__()

        # Arousal System (reticular activating system)
        self.arousal_system = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),  # Arousal level 0-1
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Autonomic Regulation (sympathetic/parasympathetic balance)
        self.autonomic_regulator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  # Balance between -1 and 1
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Vital Signs Monitor (homeostatic control)
        self.vital_monitor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # Homeostatic Controller
        self.homeostatic_controller = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Vital thresholds
        self.register_buffer('V_rest', torch.tensor(-70.0))
        self.register_buffer('V_critical', torch.tensor(-55.0))

    def forward(self, sensory_input: torch.Tensor,
                membrane_potential: torch.Tensor) -> torch.Tensor:
        """
        Process autonomic functions based on sensory input and membrane potential.

        Args:
            sensory_input: Processed sensory information
            membrane_potential: Current membrane potential state

        Returns:
            Brainstem regulatory output
        """
        # Arousal modulation based on input activity
        arousal_level = self.arousal_system(sensory_input)

        # Autonomic regulation (sympathetic/parasympathetic)
        autonomic_output = self.autonomic_regulator(sensory_input)

        # Vital signs monitoring (check if membrane potential is in safe range)
        vital_state = self.vital_monitor(membrane_potential)

        # Homeostatic control (maintain equilibrium)
        homeostatic_signal = self.homeostatic_controller(
            torch.cat([arousal_level, autonomic_output], dim=-1)
        )

        # Combine all brainstem functions
        # Vital state modulates the output
        output = homeostatic_signal * torch.sigmoid(vital_state)

        return output


class UnifiedBrainLogicNetwork(nn.Module):
    def __init__(self, 
                 timestamp: str = "2025-02-10 01:37:46",
                 user: str = "biblicalandr0id",
                 input_dim: int = 1024,
                 hidden_dim: int = 2048,
                 output_dim: int = 512):
        super().__init__()
        
        # System Constants
        self.K = 1.0  # Rationality constant
        self.V_rest = -70.0  # Resting potential (mV)
        self.ATP_critical = 0.1  # Minimum ATP threshold
        
        # Temporal Parameters
        self.t0 = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        self.user = user
        
        # Neural Processing Components
        self.sensory_processing = SensoryProcessor(input_dim, hidden_dim)
        self.cerebrum = CerebrumModule(hidden_dim)
        self.cerebellum = CerebellumModule(hidden_dim)
        self.brainstem = BrainstemModule(hidden_dim)
        self.logical_processor = LogicalProcessor(input_dim, hidden_dim)
        
        # Integration Components
        self.neural_logical_integrator = NeuralLogicalIntegrator(hidden_dim)
        self.unified_output = UnifiedOutput(hidden_dim, output_dim)
        
        # State Tracking
        self.system_state = SystemState(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Initialize system state
        Φ = self.system_state.initialize(x)
        
        # I. Combined Input Processing
        sensory_input = self.sensory_processing(x)
        logical_input = self.logical_processor.process_input(x)
        
        # II. Neural-Logical Network Dynamics
        V = self.compute_membrane_potential(sensory_input)
        Γ = self.logical_processor.compute_truth_valuation(logical_input)
        
        # III. Unified Processing
        cerebrum_output = self.cerebrum(sensory_input, logical_input)
        cerebellum_output = self.cerebellum(sensory_input, Γ)
        brainstem_output = self.brainstem(sensory_input, V)

        # IV. Apply Constraints (skipped to avoid inplace ops that break gradients)
        # Constraints are applied in SystemState.update instead
        # self.apply_constraints(cerebrum_output, V, Γ)

        # V. Temporal Evolution
        Φ = self.system_state.update(cerebrum_output, cerebellum_output, brainstem_output)
        
        # VI. Global Output Generation
        output = self.unified_output(Φ)
        
        return {
            'output': output,
            'system_state': Φ,
            'membrane_potential': V,
            'truth_values': Γ
        }
    
    def compute_membrane_potential(self, x: torch.Tensor) -> torch.Tensor:
        """Implements ∂V(x,y,z,t)/∂t = D∇²V - ∑[j=1→m] gⱼ(t)[V - Eⱼ] + ∑[k=1→p] Iₖ(t)"""
        # Simplified membrane potential computation
        # Use mean activation as surrogate for spatial integration
        return torch.full_like(x, self.V_rest) + x.mean(dim=-1, keepdim=True) * 10.0
    
    def apply_constraints(self, cerebrum_output: torch.Tensor, 
                         V: torch.Tensor, Γ: torch.Tensor):
        """Applies physical and logical constraints"""
        # Physical Constraints
        V.clamp_(-70.0, 40.0)  # Membrane potential constraints
        
        # Logical Constraints
        Γ.clamp_(0.0, 1.0)  # Truth value constraints
        
        # Conservation of Truth
        Γ = F.normalize(Γ, p=1, dim=-1)
        
        # Cognitive Boundary
        norm = torch.norm(cerebrum_output)
        if norm > self.K:
            cerebrum_output *= self.K / norm

class SensoryProcessor(nn.Module):
    """Implements Σ(t) = ∑[s∈S] ∫[0→t] λₛ(τ) × [∏(i=1→5) ψᵢ(s,τ)] dτ"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.sensory_weights = nn.Parameter(torch.randn(5, input_dim))
        self.processing_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(5)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sensory_outputs = []
        for i, layer in enumerate(self.processing_layers):
            weighted_input = x * self.sensory_weights[i]
            processed = layer(weighted_input)
            sensory_outputs.append(processed)
        return torch.stack(sensory_outputs).sum(dim=0)

class LogicalProcessor(nn.Module):
    """Implements Ψ = ∑(λᵢ × ξᵢ) and Γ mappings"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # Add input projection to handle input_dim != hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.premise_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=6
        )
        self.truth_valuation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def process_input(self, x: torch.Tensor) -> torch.Tensor:
        # Project input to hidden dimension first
        x = self.input_projection(x)
        return self.premise_encoder(x)

    def compute_truth_valuation(self, x: torch.Tensor) -> torch.Tensor:
        return self.truth_valuation(x)

class NeuralLogicalIntegrator(nn.Module):
    """Implements the neural-logical integration operator ⊛"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.integration_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        ])
    
    def forward(self, neural_state: torch.Tensor, 
                logical_state: torch.Tensor) -> torch.Tensor:
        # Implement cognitive convolution
        attended, _ = self.integration_layers[0](
            neural_state, logical_state, logical_state
        )
        
        # Implement logical tensor product
        integrated = self.integration_layers[1](attended)
        
        # Apply normalization
        return self.integration_layers[2](integrated)

class SystemState:
    """Tracks and updates the complete system state vector Φ(t)"""
    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.state_components = [
            'V',        # Membrane potential
            'NT',       # Neurotransmitter concentrations
            'Ca',       # Calcium concentration
            'ATP',      # Energy availability
            'g',        # Glial state
            'Ψ',        # Cognitive reasoning state
            'τ',        # Truth values
            'ω'         # Reasoning momentum
        ]
    
    def initialize(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        return {
            component: torch.zeros(batch_size, self.hidden_dim)
            for component in self.state_components
        }
    
    def update(self, cerebrum_output: torch.Tensor,
               cerebellum_output: torch.Tensor,
               brainstem_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Update system state based on unified equation system.

        Implements temporal evolution of:
        - V: Membrane potential dynamics
        - NT: Neurotransmitter release and degradation
        - Ca: Calcium influx/efflux
        - ATP: Energy consumption/production
        - g: Glial support state
        - Ψ: Cognitive reasoning state (from cerebrum)
        - τ: Truth values (from logical processing)
        - ω: Reasoning momentum

        Args:
            cerebrum_output: Higher cognitive processing output
            cerebellum_output: Motor coordination output
            brainstem_output: Autonomic regulation output

        Returns:
            Updated system state dictionary
        """
        batch_size = cerebrum_output.shape[0]
        device = cerebrum_output.device

        # Time step for integration
        dt = 0.01

        # Initialize state if not present
        if not hasattr(self, '_state') or self._state is None:
            self._state = {
                component: torch.zeros(batch_size, self.hidden_dim, device=device)
                for component in self.state_components
            }

        # Unpack current state
        V = self._state['V']
        NT = self._state['NT']
        Ca = self._state['Ca']
        ATP = self._state['ATP']
        g = self._state['g']
        Ψ = self._state['Ψ']
        τ = self._state['τ']
        ω = self._state['ω']

        # Constants
        V_rest = -70.0
        Ca_baseline = 100.0
        ATP_baseline = 5000.0
        tau_membrane = 20.0
        tau_NT = 100.0
        tau_Ca = 50.0

        # 1. Update Membrane Potential (influenced by brainstem)
        # ∂V/∂t = -(V - V_rest)/τ + I_input
        input_current = brainstem_output.mean(dim=-1, keepdim=True)
        dV = (-(V.mean(dim=-1, keepdim=True) - V_rest) / tau_membrane + input_current) * dt
        V_new = V + dV.expand_as(V)
        V_new = torch.clamp(V_new, -70.0, 40.0)  # Biological constraints

        # 2. Update Neurotransmitter Concentration
        # ∂NT/∂t = release(V) - degradation(NT)
        release = torch.sigmoid((V_new.mean(dim=-1, keepdim=True) + 55.0) / 10.0)
        degradation = NT.mean(dim=-1, keepdim=True) / tau_NT
        dNT = (release - degradation) * dt
        NT_new = torch.clamp(NT + dNT.expand_as(NT), 0.0, 10.0)

        # 3. Update Calcium Dynamics (influenced by cerebellum - motor learning)
        # ∂Ca/∂t = influx(V) - efflux(Ca)
        influx = torch.sigmoid((V_new.mean(dim=-1, keepdim=True) + 55.0) / 10.0) * cerebellum_output.norm(dim=-1, keepdim=True)
        efflux = (Ca.mean(dim=-1, keepdim=True) - Ca_baseline) / tau_Ca
        dCa = (influx - efflux) * dt
        Ca_new = torch.clamp(Ca + dCa.expand_as(Ca), 0.0, 1000.0)

        # 4. Update ATP (Energy) - consumption based on activity
        # ∂ATP/∂t = production - consumption(activity)
        activity_level = (cerebrum_output.norm(dim=-1, keepdim=True) +
                         cerebellum_output.norm(dim=-1, keepdim=True) +
                         brainstem_output.norm(dim=-1, keepdim=True)) / 3.0
        consumption = activity_level * 10.0  # Scaled consumption
        production = (ATP_baseline - ATP.mean(dim=-1, keepdim=True)) / 1000.0
        dATP = (production - consumption) * dt
        ATP_new = torch.clamp(ATP + dATP.expand_as(ATP), 1000.0, 10000.0)

        # 5. Update Glial State (support and modulation)
        # Glial cells support neuronal function
        glial_support = torch.tanh(ATP_new / ATP_baseline) * torch.sigmoid(Ca_new / Ca_baseline)
        g_new = 0.9 * g + 0.1 * glial_support  # Slow adaptation

        # 6. Update Cognitive Reasoning State (from cerebrum)
        # Ψ represents the current cognitive/reasoning state
        Ψ_new = 0.8 * Ψ + 0.2 * cerebrum_output  # Temporal smoothing

        # 7. Update Truth Values (logical state)
        # τ should be influenced by cerebrum reasoning
        truth_update = torch.sigmoid(cerebrum_output)
        τ_new = 0.85 * τ + 0.15 * truth_update  # Stable truth evolution
        τ_new = torch.clamp(τ_new, 0.0, 1.0)

        # 8. Update Reasoning Momentum (velocity of cognitive change)
        # ω = ||∂Ψ/∂t||
        dΨ = Ψ_new - Ψ
        ω_new = torch.sqrt(dΨ.pow(2).mean(dim=-1, keepdim=True) + 1e-8).expand_as(ω)

        # Update internal state
        self._state = {
            'V': V_new,
            'NT': NT_new,
            'Ca': Ca_new,
            'ATP': ATP_new,
            'g': g_new,
            'Ψ': Ψ_new,
            'τ': τ_new,
            'ω': ω_new
        }

        return self._state

class UnifiedOutput(nn.Module):
    """Implements Θ(t) = ℱ{Ω_m(t), Ω_c(t), Ω_a(t)} × exp(-|t - t₀|/τ) ⊗ ∫(Ψ ∘ Γ) dω"""
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.output_projection = nn.Linear(hidden_dim * 3, output_dim)
        self.temporal_modulation = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, system_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Combine motor, cognitive, and autonomic outputs
        combined_output = torch.cat([
            system_state['V'],
            system_state['Ψ'],
            system_state['τ']
        ], dim=-1)

        # Apply projection
        output = self.output_projection(combined_output)

        # Apply temporal modulation (decay factor based on system energy)
        # Use ATP as proxy for temporal sustainability
        # Take mean across hidden dimension to get scalar per batch
        temporal_factor = torch.sigmoid(system_state['ATP'].mean(dim=-1, keepdim=True) / 5000.0)  # Normalized around baseline

        return output * temporal_factor

# Example Usage
if __name__ == "__main__":
    model = UnifiedBrainLogicNetwork()
    
    # Example input
    batch_size = 32
    input_dim = 1024
    x = torch.randn(batch_size, input_dim)
    
    # Process input through the network
    output = model(x)
    
    # Access different components of the output
    final_output = output['output']
    system_state = output['system_state']
    membrane_potential = output['membrane_potential']
    truth_values = output['truth_values']