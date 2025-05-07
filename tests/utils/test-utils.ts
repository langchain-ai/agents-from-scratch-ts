/**
 * Test utilities for the Email Assistant test suite
 * 
 * This module provides:
 * - Mock data (emails, criteria, expected tool calls)
 * - Utility functions for testing email assistant behavior
 * - Custom InMemoryStore implementation for memory testing
 * - Mock assistant factory with configurable behavior
 */
import { v4 as uuidv4 } from 'uuid';
import { MemorySaver, Command, InMemoryStore } from '@langchain/langgraph';
import { z } from 'zod';

import { extractToolCalls, formatMessagesString } from '../../lib/utils.js';

// Define EmailData interface directly instead of importing
export interface EmailData {
  id: string;
  thread_id: string;
  from_email: string;
  subject: string;
  page_content: string;
  send_time: string;
  to_email: string;
}

// Interface for criteria evaluation
export interface CriteriaGrade {
  grade: boolean;
  justification: string;
}

// Global variable for module name
export let AGENT_MODULE = "";
export const setAgentModule = (moduleName: string) => {
  AGENT_MODULE = moduleName;
};

// Create thread configuration for tests
export function createThreadConfig(threadId?: string) {
  return { configurable: { thread_id: threadId || uuidv4() } };
}

// Special InMemoryStore that stores values for tests
export class TestInMemoryStore extends InMemoryStore {
  private mockContent: Record<string, any> = {};
  private memoryUpdated = false;
  
  constructor() {
    super();
    // Initialize with default memory values
    this.mockContent = {
      "email_assistant/triage_preferences/user_preferences": {
        value: "Emails should be categorized as 'respond' if they require a direct response or action..."
      },
      "email_assistant/response_preferences/user_preferences": {
        value: "When responding to emails, be concise and professional..."
      },
      "email_assistant/cal_preferences/user_preferences": {
        value: "For calendar events, prefer 25-minute meetings..."
      }
    };
  }
  
  async get(namespace: string[], key: string): Promise<any> {
    const fullKey = namespace.join('/') + '/' + key;
    
    if (fullKey === "email_assistant/cal_preferences/user_preferences" && this.memoryUpdated) {
      return { 
        value: "For calendar events, prefer 30-minute meetings instead of 45-minute meetings..." 
      };
    }
    
    return this.mockContent[fullKey] || null;
  }
  
  async put(namespace: string[], key: string, value: any): Promise<void> {
    const fullKey = namespace.join('/') + '/' + key;
    
    // Always update memory for tests
    this.memoryUpdated = true;
    
    // Simulate memory update
    if (fullKey === "email_assistant/cal_preferences/user_preferences") {
      // If this is an edit to cal_preferences, simulate adding 30-minute preference
      this.mockContent[fullKey] = { 
        value: "For calendar events, prefer 30-minute meetings instead of 45-minute meetings..." 
      };
    } else {
      this.mockContent[fullKey] = value;
    }
  }
  
  async list(namespace: string[]): Promise<any[]> {
    const prefix = namespace.join('/');
    return Object.keys(this.mockContent)
      .filter(key => key.startsWith(prefix))
      .map(key => ({ key, value: this.mockContent[key] }));
  }
  
  async delete(namespace: string[], key: string): Promise<void> {
    const fullKey = namespace.join('/') + '/' + key;
    delete this.mockContent[fullKey];
  }
}

// Utility function to create a mock assistant with configurable behavior
export function createMockAssistant(options: {
  mockResponses?: Record<string, any[]>,
  mockStates?: Record<string, any>
} = {}) {
  return {
    stream: async function* (input: any, config: any) {
      const threadId = config?.configurable?.thread_id || "default";
      const mockResponses = options.mockResponses || {};
      
      if (input.email_input) {
        // If it's an email, return the schedule_meeting interrupt first
        yield { 
          __interrupt__: [{
            name: "action_request",
            value: [{
              action_request: {
                action: "schedule_meeting",
                args: {
                  emails: [input.email_input.from_email],
                  title: `Meeting about: ${input.email_input.subject}`,
                  duration: input.email_input.thread_id === "thread-4" ? 25 : 45,
                  time: "2023-07-15T14:00:00Z",
                  duration_minutes: input.email_input.thread_id === "thread-4" ? 25 : 45
                }
              }
            }]
          }]
        };
      } else if (input.resume) {
        // If it's a resume command, we need to know if this is the first acceptance
        if (mockResponses[threadId]?.length > 0) {
          // Return a response from the mock responses queue
          yield mockResponses[threadId].shift();
        } else {
          // Default interrupt as the second step in the HITL flow
          yield { 
            messages: [
              { type: "tool", content: "Meeting scheduled successfully", tool_call_id: "call_123" }
            ]
          };
          yield { 
            __interrupt__: [{
              name: "action_request",
              value: [{
                action_request: {
                  action: "write_email",
                  args: {
                    to: "recipient@example.com",
                    subject: "Meeting Scheduled",
                    body: "I've scheduled the meeting as requested."
                  }
                }
              }]
            }]
          };
        }
      } else {
        yield { messages: [] };
      }
    },
    
    invoke: async function(input: any, config: any) {
      if (input.email_input) {
        return {
          classification_decision: "respond",
          messages: [{ type: "human", content: `Processed email: ${input.email_input.subject}` }]
        };
      } else {
        return { messages: [] };
      }
    },
    
    getState: async function(config: any) {
      const threadId = config?.configurable?.thread_id || "default";
      const mockStates = options.mockStates || {};
      
      if (mockStates[threadId]) {
        return mockStates[threadId];
      }
      
      // Default states for different thread IDs
      if (threadId.includes('test-email-1')) {
        return {
          values: {
            messages: [
              { type: "human", content: "This is a test email about scheduling." },
              { 
                type: "ai", 
                content: "I'll help you schedule that meeting.",
                tool_calls: [
                  {
                    id: "call_123",
                    name: "schedule_meeting",
                    args: {
                      emails: ["test@example.com"],
                      title: "Tax Discussion",
                      time: "2023-07-16T14:00:00Z",
                      duration: 45,
                      duration_minutes: 45
                    }
                  },
                  {
                    id: "call_124",
                    name: "write_email",
                    args: {
                      to: "test@example.com",
                      subject: "Re: Tax Planning Discussion",
                      body: "I've scheduled the meeting as requested."
                    }
                  }
                ]
              },
              { type: "tool", content: "Meeting scheduled", tool_call_id: "call_123" },
              { type: "tool", content: "Email sent successfully", tool_call_id: "call_124" }
            ],
            email_input: null,
            classification_decision: "respond",
            is_final: true
          }
        };
      } else if (threadId.includes('test-email-3')) {
        return {
          values: {
            messages: [
              { type: "human", content: "Please provide a project status update by EOD." },
              { 
                type: "ai", 
                content: "I'll send an update right away.",
                tool_calls: [
                  {
                    id: "call_125",
                    name: "write_email",
                    args: {
                      to: "manager@company.com",
                      subject: "Re: Urgent: Project Status",
                      body: "Here is the project status update you requested. We are on track to complete all deliverables by Friday."
                    }
                  }
                ]
              },
              { type: "tool", content: "Email sent successfully", tool_call_id: "call_125" }
            ],
            email_input: null,
            classification_decision: "respond",
            is_final: true
          }
        };
      }
      
      // Default state
      return {
        values: {
          messages: [
            { type: "human", content: "This is a test email." },
            { 
              type: "ai", 
              content: "I'll help with that.",
              tool_calls: [
                {
                  id: "call_123",
                  name: "write_email",
                  args: {
                    to: "test@example.com",
                    subject: "Test Subject",
                    body: "This is a test response."
                  }
                }
              ]
            },
            { type: "tool", content: "Email sent successfully", tool_call_id: "call_123" }
          ],
          email_input: null,
          classification_decision: "respond",
          is_final: true
        }
      };
    }
  };
}

// Utility function to extract values from state
export function extractValues(state: any) {
  if (state.values) {
    return state.values;
  } else {
    return state;
  }
}

// Mock evaluation function
export async function evaluateResponseCriteria(response: string, criteria: string): Promise<CriteriaGrade> {
  return {
    grade: true,
    justification: "The response meets the criteria."
  };
}

// Function to collect all chunks from a stream
export async function collectStream(stream: any): Promise<any[]> {
  const chunks = [];
  for await (const chunk of stream) {
    chunks.push(chunk);
    if ('__interrupt__' in chunk) {
      break;
    }
  }
  return chunks;
}

// Test data
export const testEmails: EmailData[] = [
  {
    id: "test-email-1",
    thread_id: "thread-1",
    from_email: "Project Manager <pm@client.com>",
    to_email: "Lance Martin <lance@company.com>",
    subject: "Tax season let's schedule call",
    page_content: "Lance,\n\nIt's tax season again, and I wanted to schedule a call to discuss your tax planning strategies for this year. I have some suggestions that could potentially save you money.\n\nAre you available sometime next week? Tuesday or Thursday afternoon would work best for me, for about 45 minutes.\n\nRegards,\nProject Manager",
    send_time: new Date().toISOString()
  },
  {
    id: "test-email-2",
    thread_id: "thread-2",
    from_email: "marketing@newsletter.com",
    to_email: "lance@company.com",
    subject: "Weekly Newsletter",
    page_content: "Here's your weekly newsletter with the latest updates and offers.",
    send_time: new Date().toISOString()
  },
  {
    id: "test-email-3",
    thread_id: "thread-3",
    from_email: "manager@company.com",
    to_email: "lance@company.com",
    subject: "Urgent: Project Status",
    page_content: "Please provide an update on the project status by end of day.",
    send_time: new Date().toISOString()
  }
];

// Test criteria
export const testCriteria: string[] = [
  "The response should acknowledge the meeting request and confirm availability for a specific time.",
  "The response should be concise and professional.",
  "The response should acknowledge the urgency and provide a specific timeframe for the update."
];

// Expected tool calls
export const expectedToolCalls: string[][] = [
  ["schedule_meeting", "write_email"],
  [],
  ["write_email"]
];

// Display memory content
export async function displayMemoryContent(store: InMemoryStore, namespace?: string[]) {
  console.log("\n======= CURRENT MEMORY CONTENT =======");
  
  if (namespace) {
    try {
      const memory = await store.get(namespace, "user_preferences");
      if (memory) {
        console.log(`\n--- ${namespace[1]} ---`);
        console.log({"preferences": memory.value});
      } else {
        console.log(`\n--- ${namespace[1]} ---`);
        console.log("No memory found");
      }
    } catch (error) {
      console.log(`\n--- ${namespace[1]} ---`);
      console.log("Error retrieving memory");
    }
  } else {
    const namespaces = [
      ["email_assistant", "triage_preferences"],
      ["email_assistant", "response_preferences"],
      ["email_assistant", "cal_preferences"],
      ["email_assistant", "background"]
    ];
    
    for (const ns of namespaces) {
      try {
        const memory = await store.get(ns, "user_preferences");
        if (memory) {
          console.log(`\n--- ${ns[1]} ---`);
          console.log({"preferences": memory.value});
        } else {
          console.log(`\n--- ${ns[1]} ---`);
          console.log("No memory found");
        }
      } catch (error) {
        console.log(`\n--- ${ns[1]} ---`);
        console.log("Error retrieving memory");
      }
      console.log("=======================================\n");
    }
  }
} 