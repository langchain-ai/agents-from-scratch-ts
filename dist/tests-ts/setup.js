import { config } from 'dotenv';
import { ChatOpenAI } from '@langchain/openai';
import { beforeAll } from '@jest/globals';
// Load environment variables
config();
// Set up global model for evaluations
global.criteriaEvalLLM = new ChatOpenAI({
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
