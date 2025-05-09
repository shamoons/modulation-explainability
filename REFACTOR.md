# Constellation Classification Refactoring Plan

This document outlines the plan to refactor the constellation classification codebase, particularly focusing on `training_constellation.py` and `validate_constellation.py`. The goal is to improve code organization, maintainability, and readability without altering functionality.

## Running Tests

To run the tests, use the following command:
```bash
uv pip install pytest  # Install pytest if not already installed
pytest  # Run all tests
```

Additional test commands:
```bash
pytest tests/validation/test_validate_constellation_contract.py  # Run specific test file
pytest -m contract  # Run tests with specific marker
pytest -v  # Run with verbose output
pytest -s  # Show print statements during test execution
```

## Goals

- Break large files (500+ lines) into smaller, focused modules
- Improve separation of concerns
- Make the code more testable
- Reduce complexity of individual functions
- Maintain all existing functionality and performance

## Phase 1: Testing Infrastructure and Initial Setup

- [x] Set up testing infrastructure
  - [x] Create a `tests/` directory with subdirectories mirroring main code structure
  - [x] Setup pytest with fixtures for common objects (models, datasets, etc.)
  - [x] Implement interface contract tests
    - [x] Test function signatures (args, returns)
    - [x] Test expected tensor shapes at key points
    - [x] Test error handling paths

- [x] Create contract tests for key functions before refactoring
  - [x] Write test for `validate_constellation` input/output shapes
  - [x] Write test for `train` function parameter handling
  - [x] Capture current behavior of curriculum stage progression
  - [x] Create "snapshot" tests of key metrics calculations

- [x] Create snapshot tests of model state during training
  - [x] Save sample input/output pairs to test for consistency
  - [x] Record confusion matrix shapes and format
  - [x] Create helper to compare model outputs before/after refactoring

## Phase 2a: Dataset and DataLoader Management

- [ ] Create `DatasetManager` class
  - [ ] Handle dataset creation and loading
  - [ ] Manage curriculum dataset updates
  - [ ] Add tests for dataset operations

- [ ] Create `DataLoaderConfig` dataclass
  - [ ] Define dataloader parameters
  - [ ] Add validation for parameters
  - [ ] Add tests for configuration

## Phase 2b: Model and Loss Management

- [ ] Create `ModelManager` class
  - [ ] Handle model initialization and loading
  - [ ] Manage model state and checkpoints
  - [ ] Add tests for model operations

- [ ] Create `LossManager` class
  - [ ] Handle loss function initialization
  - [ ] Manage dynamic loss weights
  - [ ] Add tests for loss calculations

## Phase 2c: Core Training Loop Extraction

- [ ] Create `TrainingManager` class
  - [ ] Move core training loop to class
  - [ ] Extract state management (best loss, epochs, etc.)
  - [ ] Add interface methods for validation and checkpointing
  - [ ] Add unit tests for each public method
    - [ ] Test state transitions
    - [ ] Test checkpoint creation/loading

- [ ] Create `TrainingConfig` dataclass
  - [ ] Define configuration parameters
  - [ ] Add validation for parameters
  - [ ] Add tests for configuration

## Phase 2d: Validation Loop Extraction

- [ ] Create `ValidationManager` class
  - [ ] Move validation loop to class
  - [ ] Create interface for different validation strategies
  - [ ] Handle error cases consistently
  - [ ] Add tests for tensor shapes throughout validation pipeline

- [ ] Create `ValidationConfig` dataclass
  - [ ] Define validation parameters
  - [ ] Add validation for parameters
  - [ ] Add tests for configuration

## Phase 2e: Metrics and Logging

- [ ] Create `MetricsLogger` class
  - [ ] Abstract wandb logging
  - [ ] Centralize metrics collection
  - [ ] Add consistent formatting
  - [ ] Add tests for metrics collection

- [ ] Create `TrainingMetrics` dataclass
  - [ ] Define metric structure
  - [ ] Add validation for metrics
  - [ ] Add tests for metrics

## Phase 2f: Integration and Testing

- [ ] Create `TrainingPipeline` class
  - [ ] Integrate all managers
  - [ ] Create clean interface for training
  - [ ] Add tests for pipeline integration

- [ ] Add integration tests
  - [ ] Test full training pipeline
  - [ ] Test curriculum learning integration
  - [ ] Test checkpoint saving/loading

## Phase 3: Curriculum Learning Refactoring

- [ ] Extract curriculum-specific logic to `curriculum_handler.py`
  - [ ] Move curriculum progression functions
  - [ ] Create function to update datasets
  - [ ] Create function to update loss functions
  - [ ] Add curriculum testing

- [ ] Integrate Curriculum Handler with Training Manager
  - [ ] Create clean interface
  - [ ] Make curriculum optional but well integrated

## Phase 4: Metrics and Visualization Refactoring

- [ ] Extract visualization code
  - [ ] Move confusion matrix code to `confusion_matrix.py`
  - [ ] Create visualization utilities
  - [ ] Separate data collection from plotting

## Phase 5: Training Module Refactoring

- [ ] Refactor main training loop
  - [ ] Simplify train() function 
  - [ ] Replace direct code with calls to TrainingManager
  - [ ] Extract optimizer management

- [ ] Extract batch processing logic
  - [ ] Create function for single batch processing
  - [ ] Separate forward/backward pass logic
  - [ ] Make mixed precision optional but clean

## Phase 6: Validation Module Refactoring

- [ ] Split validation functionality
  - [ ] Create `model_evaluation.py` for core validation
  - [ ] Create `metrics_calculation.py` for computing metrics
  - [ ] Create error handling utilities

- [ ] Refactor validation confusion matrix generation
  - [ ] Extract index-to-value mapping
  - [ ] Create helper for confusion matrix construction
  - [ ] Add better error handling

## Phase 7: Interface Cleanup and Integration

- [ ] Create configuration management
  - [ ] Define configuration dataclasses
  - [ ] Simplify function signatures

- [ ] Update main entry points
  - [ ] Update `train_constellation.py` to use new modules
  - [ ] Ensure backward compatibility

## Phase 8: Testing and Documentation

- [ ] Complete unit tests
  - [ ] Test each key component separately
  - [ ] Add integration tests

- [ ] Add documentation
  - [ ] Add docstrings to all modules and functions
  - [ ] Create module README files
  - [ ] Update main documentation

## Dependencies and Ordering

- Phase 1 must be completed first
- Phases 2 and 3 can be worked on in parallel
- Phase 4 depends on Phase 2
- Phases 5 and 6 depend on Phases 2, 3 and 4
- Phase 7 depends on Phases 5 and 6
- Phase 8 can be done incrementally throughout

## Suggested Testing Strategy

1. Create a benchmark training run before refactoring
2. After each phase, run the same training job and compare:
   - Loss convergence curve
   - Final accuracy metrics
   - Validation results
   - Confusion matrices
3. Automate testing where possible

## Success Criteria

- No change in model performance
- Improved code structure with no file exceeding 200 lines
- Clear separation of responsibilities between modules
- Comprehensive test coverage
- Backward compatibility preserved
