import { describe, test, expect, beforeAll } from '@jest/globals';
import { Command } from '@langchain/langgraph';

import { 
  AGENT_MODULE, 
  setAgentModule,
  setupAssistant, 
  testEmails,
  runInitialStream,
  runStreamWithCommand
} from './utils/test-utils.js';

// Set module to HITL version for these tests
setAgentModule(process.env.AGENT_MODULE || "email_assistant_hitl");

describe('HITL functionality tests', () => {
  beforeAll(() => {
    // Setup LangSmith tracing if API key is available 
    if (process.env.LANGCHAIN_API_KEY) {
      process.env.LANGCHAIN_TRACING_V2 = "true";
      process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
    }
    
    console.log(`Using agent module: ${AGENT_MODULE}`);
  });

  test('Accept write_email and schedule_meeting flow', async () => {
    // This test demonstrates the basic HITL approval flow when a user accepts all agent actions
    const email = testEmails[0]; // Meeting request email
    
    // Compile the graph
    const { emailAssistant, threadConfig } = await setupAssistant();
    
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
    
    // Accept the schedule_meeting tool call
    console.log(`\nSimulating user accepting the ${JSON.stringify(actionRequest)} tool call...`);
    const secondChunks = await runStreamWithCommand(
      emailAssistant,
      new Command({ resume: [{ type: "accept" }] }),
      threadConfig
    );
    
    // Find the next interrupt
    const secondInterrupt = secondChunks.find(chunk => '__interrupt__' in chunk);
    expect(secondInterrupt).toBeDefined();
    
    // Extract the write_email action
    const emailActionRequest = secondInterrupt?.__interrupt__[0].value[0].action_request;
    console.log("\nINTERRUPT OBJECT:");
    console.log(`Action Request: ${JSON.stringify(emailActionRequest)}`);
    
    // Verify it's a write_email request
    expect(emailActionRequest.action).toBe('write_email');
    
    // Accept the write_email tool call
    console.log(`\nSimulating user accepting the ${JSON.stringify(emailActionRequest)} tool call...`);
    const finalChunks = await runStreamWithCommand(
      emailAssistant,
      new Command({ resume: [{ type: "accept" }] }),
      threadConfig
    );
    
    // Verify completion with Done tool call
    const doneMessage = finalChunks.find(chunk => {
      if (chunk.ai_message) {
        const message = chunk.ai_message;
        return message.tool_calls && message.tool_calls.some((tc: { name: string }) => tc.name === 'Done');
      }
      return false;
    });
    
    expect(doneMessage).toBeDefined();
  }, 120000); // 2 minute timeout for LLM calls

  test('Edit tool call parameters', async () => {
    // This test demonstrates editing a tool call's parameters
    const email = testEmails[0]; // Meeting request email
    
    // Compile the graph
    const { emailAssistant, threadConfig } = await setupAssistant();
    
    // Run the graph until the first interrupt
    console.log("Running the graph until the first interrupt...");
    const initialChunks = await runInitialStream(emailAssistant, email, threadConfig);
    
    // Get the interrupt object
    const initialInterrupt = initialChunks.find(chunk => '__interrupt__' in chunk);
    expect(initialInterrupt).toBeDefined();
    
    // Extract the action request from the interrupt
    const actionRequest = initialInterrupt?.__interrupt__[0].value[0].action_request;
    
    // Verify it's a schedule_meeting request
    expect(actionRequest.action).toBe('schedule_meeting');
    
    // Edit the meeting duration and time
    const editedArgs = {
      ...actionRequest.args,
      duration_minutes: 30, // Change from 45 minutes to 30 minutes
      start_time: 15 // Change from 2pm to 3pm
    };
    
    // Edit the tool call
    console.log(`\nSimulating user editing the meeting parameters...`);
    const secondChunks = await runStreamWithCommand(
      emailAssistant,
      new Command({ resume: [{ type: "edit", args: editedArgs }] }),
      threadConfig
    );
    
    // Find the next interrupt for write_email
    const secondInterrupt = secondChunks.find(chunk => '__interrupt__' in chunk);
    expect(secondInterrupt).toBeDefined();
    
    // Extract the write_email action
    const emailActionRequest = secondInterrupt?.__interrupt__[0].value[0].action_request;
    
    // Verify it's a write_email request
    expect(emailActionRequest.action).toBe('write_email');
    
    // Verify email mentions the edited parameters (30 minutes, 3pm)
    const emailContent = emailActionRequest.args.content;
    expect(emailContent).toContain('30');
    expect(emailContent).toContain('3:00 PM');
    
    // Accept the write_email
    const finalChunks = await runStreamWithCommand(
      emailAssistant,
      new Command({ resume: [{ type: "accept" }] }),
      threadConfig
    );
    
    // Verify completion
    const state = await emailAssistant.getState(threadConfig);
    expect(state.is_final).toBeTruthy();
  }, 120000); // 2 minute timeout for LLM calls

  test('Reject tool call with feedback', async () => {
    // This test demonstrates rejecting a tool call with feedback
    const email = testEmails[0]; // Meeting request email
    
    // Compile the graph
    const { emailAssistant, threadConfig } = await setupAssistant();
    
    // Run the graph until the first interrupt
    console.log("Running the graph until the first interrupt...");
    const initialChunks = await runInitialStream(emailAssistant, email, threadConfig);
    
    // Get the interrupt object
    const initialInterrupt = initialChunks.find(chunk => '__interrupt__' in chunk);
    expect(initialInterrupt).toBeDefined();
    
    // Extract the action request from the interrupt
    const actionRequest = initialInterrupt?.__interrupt__[0].value[0].action_request;
    
    // Verify it's a schedule_meeting request
    expect(actionRequest.action).toBe('schedule_meeting');
    
    // Reject the tool call with feedback
    console.log(`\nSimulating user rejecting the tool call with feedback...`);
    const secondChunks = await runStreamWithCommand(
      emailAssistant,
      new Command({ resume: [{ type: "reject", args: "I'm not available next week. Please suggest the following week instead." }] }),
      threadConfig
    );
    
    // The agent should now propose a different meeting time
    // Find the next interrupt
    const secondInterrupt = secondChunks.find(chunk => '__interrupt__' in chunk);
    expect(secondInterrupt).toBeDefined();
    
    // Extract the action request
    const newActionRequest = secondInterrupt?.__interrupt__[0].value[0].action_request;
    
    // Should still be a schedule_meeting but with a different date
    expect(newActionRequest.action).toBe('schedule_meeting');
    
    // The new proposed date should be in a different week
    const originalDate = new Date(actionRequest.args.preferred_day);
    const newDate = new Date(newActionRequest.args.preferred_day);
    
    // Calculate the difference in days
    const dayDifference = Math.floor((newDate.getTime() - originalDate.getTime()) / (1000 * 60 * 60 * 24));
    
    // Expect at least 7 days difference (next week)
    expect(dayDifference).toBeGreaterThanOrEqual(7);
  }, 120000); // 2 minute timeout for LLM calls
}); 