import pennylane as qml
import inspect

# Print function signature of dm_from_state_vector
print("Function signature of dm_from_state_vector:")
print(inspect.signature(qml.math.dm_from_state_vector))

# Print docstring
print("\nDocstring:")
print(qml.math.dm_from_state_vector.__doc__) 