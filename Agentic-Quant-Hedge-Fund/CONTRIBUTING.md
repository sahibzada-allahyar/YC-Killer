# Contributing to Agentic Quantitative Trading

We welcome contributions from the community! This document provides guidelines for contributing to the Agentic Quantitative Trading system.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We are committed to providing a welcoming and inspiring community for all.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs
- Include detailed information about your environment
- Provide steps to reproduce the issue
- Include relevant error messages and stack traces

### Suggesting Features

- Open a GitHub issue with the "enhancement" label
- Clearly describe the feature and its use case
- Explain why this feature would be beneficial to the project

### Contributing Code

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/YC-Killer.git
   cd YC-Killer/Agentic-Quant-Hedge-Fund
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up the development environment**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Make your changes**
   - Follow the existing code style and conventions
   - Add tests for new functionality
   - Update documentation as needed

5. **Run tests**
   ```bash
   python -m pytest src/agentic_quant/tests/
   ```

6. **Run linting and formatting**
   ```bash
   black src/agentic_quant/
   mypy src/agentic_quant/
   ```

7. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

8. **Push to your fork and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 Python style guide
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Use meaningful variable and function names

### Testing

- Write unit tests for all new functionality
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage
- Use pytest for testing framework

### Documentation

- Update README.md if your changes affect usage
- Add docstrings to new functions and classes
- Include type hints and parameter descriptions
- Update API documentation for public interfaces

### Mathematical Transforms

When adding new mathematical transforms:

1. **Register the transform** in `core/transforms.py`:
   ```python
   @_register("your_transform", (arity_min, arity_max))
   @nb.njit(cache=True, fastmath=True)
   def your_transform(x, *args):
       # Implementation here
       return result
   ```

2. **Add tests** in `tests/unit/test_transforms.py`:
   ```python
   def test_your_transform():
       """Test your transform function."""
       transform_fn = get("your_transform")
       # Test cases here
   ```

3. **Document the transform** with clear mathematical description

### LLM Integration

When modifying LLM-related code:

- Ensure prompts are well-structured and tested
- Handle API errors gracefully
- Consider rate limiting and costs
- Test with different models when possible

## Pull Request Process

1. **Ensure your PR has a clear title and description**
2. **Reference any related issues**
3. **Include screenshots or examples if applicable**
4. **Ensure all tests pass**
5. **Wait for code review from maintainers**
6. **Address any requested changes**

## Getting Help

- Join our community discussions
- Ask questions in GitHub issues
- Contact the maintainers directly: sahibzada@singularityresearchlabs.com

## Recognition

Contributors will be recognized in:
- The project's contributor list
- Release notes for significant contributions
- Our team page for exceptional contributions

Thank you for contributing to democratizing quantitative finance!

