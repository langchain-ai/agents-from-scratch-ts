# LangGraph Email Agent Testing Suite

This is a TypeScript testing framework for evaluating LangGraph agents, particularly email processing assistants with human-in-the-loop (HITL) and memory capabilities.

## Overview

The testing suite mirrors the Python-based testing framework but adapted for TypeScript and Jest. It includes:

1. **Response Quality Testing**: Tests the agent's ability to generate quality email responses.
2. **Tool Call Testing**: Verifies the agent makes the expected tool calls.
3. **HITL Workflow Testing**: Tests the human-in-the-loop workflow with accept/edit/reject operations.
4. **Memory Testing**: Verifies memory persistence and preference learning across conversations.

## Installation

```bash
# Install dependencies
npm install
```

## Running Tests

```bash
# Run from the project root directory

# Run all tests
pnpm test

# Run specific test suites
pnpm test:hitl     # Run HITL workflow tests
pnpm test:memory   # Run memory functionality tests
pnpm test:base     # Run basic email assistant tests

# Run with specific module
AGENT_MODULE=email_assistant_hitl_memory pnpm test
```

## Test Files

- `test_response.test.ts`: Tests email response quality and tool calling accuracy
- `hitl_testing.test.ts`: Tests human-in-the-loop workflows
- `memory_testing.test.ts`: Tests memory persistence and learning

## ESM Module Setup

This project uses ESM modules which required specific Jest configuration:

1. The setup file is in `.mjs` format to support ESM modules
2. Jest is configured to use the proper ESM preset 
3. Global types are declared in a separate `.d.ts` file

## Mock Assistant Implementation

The testing suite uses a configurable mock assistant approach:

- `createMockAssistant()` in `test-utils.ts` creates a mock assistant with customizable responses
- Thread-specific responses can be provided via `mockResponses` and `mockStates` parameters
- This approach allows tests to be tailored for different test scenarios without complex mocks

## Memory Testing

Memory tests use a specialized `TestInMemoryStore` that:

1. Simulates memory storage and retrieval
2. Tracks memory changes
3. Provides methods for displaying memory content

## Environment Variables

Create a `.env` file in the root directory with:

```
OPENAI_API_KEY=your-api-key

# Optional: For tracing
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_CALLBACKS_BACKGROUND=true
```

## Maintenance and Extension

### Adding New Tests

To add new tests:

1. Use the existing test pattern for consistency
2. Leverage the configurability of `createMockAssistant()` for custom scenarios
3. Add test-specific mock responses and states as needed

### Updating Mocks

To update the mock assistant behavior:

1. Modify the `createMockAssistant()` function in `test-utils.ts`
2. Add new mock states for specific thread scenarios
3. Extend the function to handle new interrupt types or responses

### Debugging Tests

For test debugging:

1. Look at the console logs for the streams, interrupts, and memory content
2. Check assertions for the expected vs actual values
3. Verify the mock responses and states match the expected agent behavior

### Extending Test Utilities

To add new utility functions:

1. Add the function to `test-utils.ts`
2. Document the purpose and usage
3. Export the function for use in test files

## Notes

- Tests may take longer to run due to LLM calls
- Default timeout is set to 2 minutes for LLM-based tests 
- The mock assistant approach allows for faster tests without actual LLM calls 