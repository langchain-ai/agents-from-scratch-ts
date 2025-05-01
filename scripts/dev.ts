import 'dotenv/config';
import { getEmailAssistant } from './email_assistant.js';
import { EmailData } from '../lib/schemas.js';

async function main() {
  console.log('Starting email assistant...');
  
  try {
    // Initialize the email assistant
    const agent = await getEmailAssistant();
    
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
    console.log('Processing email...');
    const result = await agent.invoke({
      email_input: exampleEmail
    });
    
    console.log('Processing complete!');
    console.log('Result:', JSON.stringify(result, null, 2));
  } catch (error) {
    console.error('Error running email assistant:', error);
  }
}

main().catch(console.error); 