import { config } from 'dotenv';
import { ChatOpenAI } from '@langchain/openai';
import { beforeAll } from '@jest/globals';

// Load environment variables
config();

// Set up global model for evaluations
// @ts-ignore - Global types will be picked up from setup.d.ts
globalThis.criteriaEvalLLM = new ChatOpenAI({
  modelName: 'gpt-4o',
  temperature: 0
});

// This runs before all tests
beforeAll(() => {
  // Enable LangSmith tracing if credentials are available
  if (process.env.LANGCHAIN_API_KEY) {
    process.env.LANGCHAIN_TRACING_V2 = "true";
    process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
  }
}); 