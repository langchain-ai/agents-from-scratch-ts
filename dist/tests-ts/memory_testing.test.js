import { describe, test, expect, beforeAll } from '@jest/globals';
import { Command } from '@langchain/langgraph';
import { AGENT_MODULE, setAgentModule, setupAssistant, testEmails, runInitialStream, runStreamWithCommand, displayMemoryContent } from './utils/test-utils.js';
// Set module to HITL+Memory version for these tests
setAgentModule(process.env.AGENT_MODULE || "email_assistant_hitl_memory");
describe('Memory functionality tests', () => {
    beforeAll(() => {
        // Setup LangSmith tracing if API key is available 
        if (process.env.LANGCHAIN_API_KEY) {
            process.env.LANGCHAIN_TRACING_V2 = "true";
            process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
        }
        console.log(`Using agent module: ${AGENT_MODULE}`);
    });
    test('Accept flow without memory updates', async () => {
        // This test demonstrates how accepting without feedback doesn't update memory
        const email = testEmails[0]; // Meeting request email
        // Compile the graph
        const { emailAssistant, threadConfig, store } = await setupAssistant();
        // Check initial memory state
        await displayMemoryContent(store);
        // Run the graph until the first interrupt
        console.log("Running the graph until the first interrupt...");
        const initialChunks = await runInitialStream(emailAssistant, email, threadConfig);
        // Get the interrupt object
        const initialInterrupt = initialChunks.find(chunk => '__interrupt__' in chunk);
        expect(initialInterrupt).toBeDefined();
        // Extract the action request from the interrupt
        const actionRequest = initialInterrupt?.__interrupt__[0].value[0].action_request;
        console.log("\nINTERRUPT OBJECT:");
        console.log(`Action Request: ${JSON.stringify(actionRequest)}`);
        // Verify it's a schedule_meeting request
        expect(actionRequest.action).toBe('schedule_meeting');
        // Get initial calendar preferences
        const initialCalPreferences = await store.get(["email_assistant", "cal_preferences"], "user_preferences");
        const initialPrefsContent = initialCalPreferences?.value;
        // Accept without modification
        console.log(`\nSimulating user accepting the ${actionRequest.action} tool call...`);
        const secondChunks = await runStreamWithCommand(emailAssistant, new Command({ resume: [{ type: "accept" }] }), threadConfig);
        // Find the next interrupt
        const secondInterrupt = secondChunks.find(chunk => '__interrupt__' in chunk);
        expect(secondInterrupt).toBeDefined();
        // Extract the write_email action
        const emailActionRequest = secondInterrupt?.__interrupt__[0].value[0].action_request;
        // Verify no memory changes after simple accept
        const currentCalPreferences = await store.get(["email_assistant", "cal_preferences"], "user_preferences");
        expect(currentCalPreferences?.value).toEqual(initialPrefsContent);
        // Accept the write_email tool call
        await runStreamWithCommand(emailAssistant, new Command({ resume: [{ type: "accept" }] }), threadConfig);
        // Verify memory still unchanged
        const finalCalPreferences = await store.get(["email_assistant", "cal_preferences"], "user_preferences");
        expect(finalCalPreferences?.value).toEqual(initialPrefsContent);
    }, 120000); // 2 minute timeout for LLM calls
    test('Memory updates based on edit with feedback', async () => {
        // This test demonstrates how editing with feedback updates memory
        const email = testEmails[0]; // Meeting request email
        // Compile the graph
        const { emailAssistant, threadConfig, store } = await setupAssistant();
        // Check initial memory state
        await displayMemoryContent(store);
        // Run the graph until the first interrupt
        console.log("Running the graph until the first interrupt...");
        const initialChunks = await runInitialStream(emailAssistant, email, threadConfig);
        // Get the interrupt object
        const initialInterrupt = initialChunks.find(chunk => '__interrupt__' in chunk);
        expect(initialInterrupt).toBeDefined();
        // Extract the action request from the interrupt
        const actionRequest = initialInterrupt?.__interrupt__[0].value[0].action_request;
        // Get initial calendar preferences
        const initialCalPreferences = await store.get(["email_assistant", "cal_preferences"], "user_preferences");
        const initialPrefsContent = initialCalPreferences?.value;
        // Edit the meeting duration and add explicit feedback
        const editedArgs = {
            ...actionRequest.args,
            duration_minutes: 30, // Change from 45 to 30 minutes
        };
        // Edit with feedback about preference
        console.log(`\nSimulating user editing with feedback about 30-minute meeting preference...`);
        const secondChunks = await runStreamWithCommand(emailAssistant, new Command({ resume: [{
                    type: "edit",
                    args: editedArgs,
                    feedback: "I always prefer 30-minute meetings unless longer is specifically needed."
                }] }), threadConfig);
        // Check memory after edit with feedback
        const updatedCalPreferences = await store.get(["email_assistant", "cal_preferences"], "user_preferences");
        const updatedPrefsContent = updatedCalPreferences?.value;
        // Verify memory was updated with 30-minute preference
        expect(updatedPrefsContent).not.toEqual(initialPrefsContent);
        expect(updatedPrefsContent).toContain("30-minute");
        // Finish the flow by accepting the email
        const secondInterrupt = secondChunks.find(chunk => '__interrupt__' in chunk);
        expect(secondInterrupt).toBeDefined();
        await runStreamWithCommand(emailAssistant, new Command({ resume: [{ type: "accept" }] }), threadConfig);
    }, 120000); // 2 minute timeout for LLM calls
    test('Memory affects subsequent emails', async () => {
        // This test demonstrates how memory affects future interactions
        const email = testEmails[0]; // Meeting request email
        // Compile the graph
        const { emailAssistant, store } = await setupAssistant();
        // Update the calendar preferences directly to set a known state
        await store.put(["email_assistant", "cal_preferences"], "user_preferences", { value: "I strictly prefer 25-minute meetings. This is a non-negotiable preference." });
        // Create a new email with different times for second test
        const newEmail = {
            ...testEmails[0],
            id: "test-email-4",
            thread_id: "thread-4",
            subject: "Another meeting request",
            page_content: "Lance,\n\nCan we schedule a 45-minute call next Monday?\n\nRegards,\nSomeone"
        };
        // Create a new thread config for this test
        const newThreadConfig = { configurable: { thread_id: "new-test-thread" } };
        // Run the graph until the first interrupt
        console.log("Processing new email with existing memory preferences...");
        const initialChunks = await runInitialStream(emailAssistant, newEmail, newThreadConfig);
        // Get the interrupt object
        const initialInterrupt = initialChunks.find(chunk => '__interrupt__' in chunk);
        expect(initialInterrupt).toBeDefined();
        // Extract the action request from the interrupt
        const actionRequest = initialInterrupt?.__interrupt__[0].value[0].action_request;
        // Verify the scheduler honors the 25-minute preference from memory
        expect(actionRequest.action).toBe('schedule_meeting');
        expect(actionRequest.args.duration_minutes).toBe(25);
        // Verify the tool call proposal mentions the 25-minute preference
        console.log(`\nVerifying memory is used in the proposal: ${JSON.stringify(actionRequest)}`);
    }, 120000); // 2 minute timeout for LLM calls
});
