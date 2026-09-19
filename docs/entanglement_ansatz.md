# Entanglement-Promoting Ansatz Options

This document describes the new ansatz options added to QNN Explorer that are designed to promote high entanglement in quantum circuits.

## Why Entanglement Matters

Entanglement is a key quantum resource that enables quantum algorithms to potentially outperform classical algorithms. When training hybrid quantum-classical models:

- Higher entanglement generally leads to more expressive quantum circuits
- More expressive circuits can represent more complex functions
- Complex quantum correlations may enable the model to learn patterns that are difficult for classical models

## Available Entanglement-Promoting Ansatzes

### 1. Hardware Efficient Ansatz

**Entanglement Potential: High**

This ansatz is designed to use gates that are typically efficient to implement on quantum hardware while maximizing entanglement through all-to-all connectivity.

```
for each layer:
    # Rotation layer (3 params per qubit)
    Apply RX, RY, RZ to each qubit
    
    # Entanglement layer - all-to-all connectivity
    Apply CZ between every pair of qubits
```

**Key features:**
- Uses all three rotation gates (RX, RY, RZ) for maximum expressivity
- All-to-all connectivity maximizes entanglement potential
- Uses CZ gates which are often more efficient on hardware than CNOT

**Parameter count:** 3 × qubits × layers

### 2. GHZ-type Ansatz

**Entanglement Potential: High**

This ansatz creates entanglement similar to GHZ states (|000...0〉 + |111...1〉), which are maximally entangled states across all qubits.

```
# Initial layer
Apply Hadamard to first qubit

for each layer:
    # Entanglement layer - star topology
    Apply CNOT from first qubit to all other qubits
    
    # Rotation layer (3 params per qubit)
    Apply RX, RY, RZ to each qubit
```

**Key features:**
- Creates GHZ-like entanglement where all qubits are correlated
- Star topology efficiently creates multi-qubit entanglement with minimal gate count
- Highly efficient for creating large-scale entanglement

**Parameter count:** 3 × qubits × layers

### 3. Brickwall Ansatz

**Entanglement Potential: Medium to High**

This alternates between even and odd qubit pairs for entanglement, creating a pattern that efficiently entangles all qubits.

```
for each layer:
    # Rotation layer (2 params per qubit)
    Apply RY, RZ to each qubit
    
    # Even layer: entangle qubits (0,1), (2,3), ...
    if layer % 2 == 0:
        Apply CNOT between qubits (0,1), (2,3), ...
    # Odd layer: entangle qubits (1,2), (3,4), ...
    else:
        Apply CNOT between qubits (1,2), (3,4), ...
        
    # Final layer: connect the ends for circular entanglement
    Apply CNOT between last and first qubit
```

**Key features:**
- Efficient entanglement with linear number of gates
- Alternating pattern creates entanglement between all qubits over multiple layers
- Similar to patterns used in quantum supremacy circuits

**Parameter count:** 2 × qubits × layers

### 4. Quantum Volume Ansatz

**Entanglement Potential: Very High**

Based on IBM's Quantum Volume circuit design, this ansatz implements a pattern that maximizes entanglement between all pairs of qubits.

```
for each layer:
    # Rotation layer (3 params per qubit)
    Apply RZ, RY, RZ to each qubit (full SU(2) rotations)
    
    # Entanglement layer - layer-dependent pairing
    For each pair i, pair qubits based on permutation:
        q1 = (i*2 + layer) % num_qubits
        q2 = (i*2 + 1 + layer) % num_qubits
        Apply CNOT between q1 and q2
```

**Key features:**
- Uses full SU(2) single-qubit rotations (RZ-RY-RZ sequence)
- Layer-dependent qubit pairing ensures all qubit pairs interact over multiple layers
- Designed specifically for high entanglement generation
- Based on circuits used for quantum computer benchmarking

**Parameter count:** 3 × qubits × layers

## Comparison and Selection Guidelines

| Ansatz | Parameters | Circuit Depth | Entanglement | Best Use Case |
|--------|------------|---------------|--------------|---------------|
| Hardware Efficient | 3 × N × L | Medium-High | High | When connectivity is not limited |
| GHZ-type | 3 × N × L | Low | High | When star connectivity is available |
| Brickwall | 2 × N × L | Medium | Medium-High | When only nearest-neighbor connectivity is available |
| Quantum Volume | 3 × N × L | Medium | Very High | For maximum entanglement with arbitrary connectivity |

## Usage in QNN Explorer

To use these ansatzes in QNN Explorer:

1. In the **Model Configuration** tab, select one of the new ansatz options from the "Circuit Ansatz" dropdown
2. Recommended starting settings for high entanglement:
   - 4-6 qubits
   - 2-4 layers
   - "State Vector" observation type to preserve quantum correlations

## Technical Details

All ansatzes have been implemented in `core/quantum_models.py` and are available through the AVAILABLE_ANSATZES dictionary in `core/fusion.py`. 