# Contributing to Jarvis Executive Assistant

First off, thank you for considering contributing to Jarvis Executive Assistant! It's people like you that make this project such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include screenshots and animated GIFs if possible
* Include error messages and stack traces
* Include the version of the project you're using

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Explain why this enhancement would be useful
* List some other applications where this enhancement exists, if applicable

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Follow the TypeScript styleguide
* Include screenshots and animated GIFs in your pull request whenever possible
* Document new code
* End all files with a newline

## Development Process

1. Fork the repo and create your branch from \`main\`
2. Run \`npm install\` in the root directory
3. Make your changes
4. Add tests if applicable
5. Run the test suite with \`npm test\`
6. Make sure your code lints with \`npm run lint\`
7. Issue that pull request!

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line
* Consider starting the commit message with an applicable emoji:
    * 🎨 `:art:` when improving the format/structure of the code
    * 🐎 `:racehorse:` when improving performance
    * 🚱 `:non-potable_water:` when plugging memory leaks
    * 📝 `:memo:` when writing docs
    * 🐛 `:bug:` when fixing a bug
    * 🔥 `:fire:` when removing code or files
    * 💚 `:green_heart:` when fixing the CI build
    * ✅ `:white_check_mark:` when adding tests
    * 🔒 `:lock:` when dealing with security
    * ⬆️ `:arrow_up:` when upgrading dependencies
    * ⬇️ `:arrow_down:` when downgrading dependencies

## Styleguides

### TypeScript Styleguide

* Use TypeScript strict mode
* Use interface over type when possible
* Document complex types
* Use meaningful variable names
* Follow the existing code style

### Documentation Styleguide

* Use [Markdown](https://guides.github.com/features/mastering-markdown/)
* Reference methods and classes in markdown with the custom \`{}\` notation:
    * Reference classes with \`{ClassName}\`
    * Reference instance methods with \`{ClassName#methodName}\`
    * Reference class methods with \`{ClassName.methodName}\`

## Project Structure

The project follows a monorepo structure with three main packages:

* \`packages/frontend\`: React frontend application
* \`packages/backend\`: Node.js backend server
* \`packages/shared\`: Shared types and utilities

### Frontend Guidelines

* Components should be functional and use hooks
* Use Material-UI components when possible
* Keep components small and focused
* Use TypeScript types for props
* Test components with React Testing Library

### Backend Guidelines

* Follow RESTful API design principles
* Document API endpoints with JSDoc comments
* Use dependency injection when possible
* Write unit tests for services
* Handle errors appropriately

## Testing

* Write tests for new features
* Update tests when modifying features
* Ensure all tests pass before submitting PR
* Aim for good test coverage

## Additional Notes

### Issue and Pull Request Labels

* \`bug\`: Something isn't working
* \`enhancement\`: New feature or request
* \`documentation\`: Improvements or additions to documentation
* \`good first issue\`: Good for newcomers
* \`help wanted\`: Extra attention is needed
* \`question\`: Further information is requested

## Getting Help

If you need help, you can:

* Check the documentation
* Join our Discord community
* Open an issue with the question label
* Email sahibzada@singularityresearchlabs.com

Thank you for contributing to Jarvis Executive Assistant! 🎉
