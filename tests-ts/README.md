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

## Utils

The testing suite includes utility functions in `utils/test-utils.ts` for:

- Setting up the assistant for different implementations
- Running streams with interrupt handling
- Evaluating response quality with LLMs
- Displaying memory content

## Environment Variables

Create a `.env` file in the root directory with:

```
OPENAI_API_KEY=your-api-key

# Optional: For tracing
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_CALLBACKS_BACKGROUND=true
```

## Notes

- Tests may take longer to run due to LLM calls
- Default timeout is set to 2 minutes for LLM-based tests 