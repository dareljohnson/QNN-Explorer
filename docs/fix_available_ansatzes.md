# Fix for AVAILABLE_ANSATZES Import Error

## Issue Description

The application was failing with the following error:

```
ImportError: cannot import name 'AVAILABLE_ANSATZES' from 'core.fusion' (C:\development\python_apps\quantum_qnn2\core\fusion.py)
Traceback:
File "C:\development\python_apps\quantum_qnn2\app.py", line 21, in <module>
    from core.fusion import HybridModel, AVAILABLE_ANSATZES
```

This error occurred because `app.py` was attempting to import `AVAILABLE_ANSATZES` from the `core.fusion` module, but the variable was not defined in that file. The variable was previously defined in `app.py` itself but had been commented out during a refactoring, and the import statement was added without adding the corresponding definition to the `core.fusion` module.

## Solution Implemented

The solution was to define the `AVAILABLE_ANSATZES` dictionary in `core.fusion.py` as expected by `app.py`. This was done by:

1. Adding the dictionary definition at the top of `core/fusion.py` after the imports:

```python
# Define available ansatzes that can be used in the quantum circuit
AVAILABLE_ANSATZES = {
    "Ansatz 1 (Rot+CNOT Chain)": ansatz1,
    # Add more ansatzes as they become available
    # "Ansatz 2 (Rot+CZ AllPairs)": None,  # To be implemented later
}
```

This dictionary maps human-readable names of quantum circuit ansatzes to their corresponding functions. Currently, it only includes one ansatz function, `ansatz1`, which is imported from the `core.quantum_models` module.

## Testing

A dedicated test file (`tests/test_ansatzes_import.py`) was created to verify the fix. The test ensures that:

1. `AVAILABLE_ANSATZES` can be successfully imported from `core.fusion`
2. The dictionary is properly structured and contains the expected ansatz
3. The ansatz function in the dictionary can be called correctly with appropriate parameters

The tests pass successfully, confirming that the fix resolves the import error.

## Why This Fix Works

The fix works because it:

1. Aligns the code structure with the import statement in `app.py`
2. Places the `AVAILABLE_ANSATZES` dictionary in a logical location - the same module as the `HybridModel` class that uses it
3. Makes the code more maintainable by centralizing the ansatz definitions

## Future Considerations

As new ansatz functions are developed, they should be:

1. Implemented in the `core.quantum_models` module
2. Imported into `core.fusion`
3. Added to the `AVAILABLE_ANSATZES` dictionary with appropriate descriptive names

This structure allows for a clean separation of concerns between:
- The implementation of quantum circuit ansatzes (`core.quantum_models`)
- The registration of available ansatzes for use in the UI (`core.fusion`)
- The UI code that presents these options to users (`app.py`) 