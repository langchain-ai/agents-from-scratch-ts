import { describe, test, expect, beforeAll } from '@jest/globals';
import { AGENT_MODULE, setAgentModule, setupAssistant, extractValues, processStream, testEmails, testCriteria, expectedToolCalls, evaluateResponseCriteria } from './utils/test-utils.js';
import { extractToolCalls, formatMessagesString } from '../lib/utils.js';
// Python uses command-line arguments with pytest, we'll use an environment variable
// or default to the HITL+Memory version
setAgentModule(process.env.AGENT_MODULE || "email_assistant_hitl_memory");
describe('Email response tests', () => {
    beforeAll(() => {
        // Setup LangSmith tracing if API key is available 
        if (process.env.LANGCHAIN_API_KEY) {
            process.env.LANGCHAIN_TRACING_V2 = "true";
            process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
        }
        console.log(`Using agent module: ${AGENT_MODULE}`);
    });
    describe('Tool call tests', () => {
        // Only include emails that should have tool calls (triage_output == "respond")
        const responseCases = testEmails
            .map((email, i) => [email, testCriteria[i], expectedToolCalls[i]])
            .filter((_, i) => expectedToolCalls[i].length > 0);
        test.each(responseCases)('processes %s with expected tool calls', async (emailInput, criteria, expectedCalls) => {
            // Log test info
            console.log(`Processing ${emailInput.subject}...`);
            // Set up the assistant
            const { emailAssistant, threadConfig } = await setupAssistant();
            // Run the agent with HITL handling
            await processStream(emailAssistant, { "email_input": emailInput }, threadConfig);
            // Get the final state
            const state = await emailAssistant.getState(threadConfig);
            const values = extractValues(state);
            // Extract tool calls from messages
            const extractedToolCalls = extractToolCalls(values.messages);
            // Check if all expected tool calls are in the extracted ones
            const missingCalls = expectedCalls.filter((call) => !extractedToolCalls.includes(call.toLowerCase()));
            // Extra calls are allowed (we only fail if expected calls are missing)
            const extraCalls = extractedToolCalls.filter((call) => !expectedCalls.map((c) => c.toLowerCase()).includes(call.toLowerCase()));
            // Log for debugging
            console.log('Extracted tool calls:', extractedToolCalls);
            console.log('Missing calls:', missingCalls);
            console.log('Extra calls:', extraCalls);
            // Get formatted messages for detailed logging
            const allMessagesStr = formatMessagesString(values.messages);
            console.log('Response:', allMessagesStr);
            // Assert that there are no missing calls
            expect(missingCalls.length).toBe(0);
        }, 60000 // 60 second timeout for LLM calls
        );
    });
    describe('Response quality tests', () => {
        // Only test emails that require a response
        const responseCases = testEmails
            .map((email, i) => [email, testCriteria[i], expectedToolCalls[i]])
            .filter((_, i) => expectedToolCalls[i].length > 0);
        test.each(responseCases)('produces quality response for %s', async (emailInput, criteria, expectedCalls) => {
            // Log test info
            console.log(`Processing ${emailInput.subject}...`);
            // Set up the assistant
            const { emailAssistant, threadConfig } = await setupAssistant();
            // Run the agent with HITL handling
            await processStream(emailAssistant, { "email_input": emailInput }, threadConfig);
            // Get the final state
            const state = await emailAssistant.getState(threadConfig);
            const values = extractValues(state);
            // Format all messages for evaluation
            const allMessagesStr = formatMessagesString(values.messages);
            // Evaluate the response against criteria
            const evaluation = await evaluateResponseCriteria(allMessagesStr, criteria);
            // Log the evaluation
            console.log('Evaluation:', evaluation);
            // Assert that the response meets the criteria
            expect(evaluation.grade).toBe(true);
        }, 60000 // 60 second timeout for LLM calls
        );
    });
});
