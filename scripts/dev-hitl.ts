import 'dotenv/config';
import { getHitlEmailAssistant } from './email_assistant_hitl.js';
import { EmailData } from '../lib/schemas.js';

async function main() {
  console.log('Starting HITL email assistant...');
  
  try {
    // Initialize the HITL email assistant
    const agent = await getHitlEmailAssistant();
    
    // Example email data
    const exampleEmail: EmailData = {
      id: "123456",
      thread_id: "thread_123456",
      from_email: "example@example.com",
      to_email: "assistant@yourcompany.com",
      subject: "Meeting Request Tomorrow",
      page_content: "Hi there,\n\nI was wondering if we could schedule a meeting tomorrow to discuss the project.\n\nThanks,\nExample User",
      send_time: new Date().toISOString()
    };
    
    // Invoke the agent with example data
    console.log('Processing email with human in the loop...');
    const result = await agent.invoke({
      email_input: exampleEmail
    });
    
    console.log('Processing complete!');
    console.log('Result:', JSON.stringify(result, null, 2));
  } catch (error) {
    console.error('Error running HITL email assistant:', error);
  }
}

main().catch(console.error); 