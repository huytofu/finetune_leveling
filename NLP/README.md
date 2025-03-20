## Testing

The library includes comprehensive tests for all components, ensuring functionality and reliability. The test suite covers:

- Integration tests for the complete training pipeline
- Unit tests for individual components, including:
  - Checkpoint Manager
  - Quantization Manager
  - Type Utilities
  - PEFT adapters and utilities

### Running Tests

To run all tests:

```bash
cd explore_llm/NLP
python tests/run_tests.py
```

To run a specific test file:

```bash
python -m unittest explore_llm.NLP.tests.test_checkpoint_manager
python -m unittest explore_llm.NLP.tests.test_quantization_manager
python -m unittest explore_llm.NLP.tests.test_type_utils
python -m unittest explore_llm.NLP.tests.test_integration
```

For more details about Parameter-Efficient Fine-Tuning (PEFT) implementation, see [README_PEFT.md](README_PEFT.md). 