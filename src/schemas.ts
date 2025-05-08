import { z } from "zod";
import { BaseMessage } from "@langchain/core/messages";
import "@langchain/langgraph/zod";
import { addMessages, Messages, StateGraph } from "@langchain/langgraph";
// TO DO REPLACE ANY WITH THE CORRECT TYPE ask David/Ben

// Define the Zod schemas for the email assistant states
export const BaseEmailAgentState = z.object({
  messages: z
    .custom<Messages>() // Using any to support all Message types
    .default(() => [])
    .langgraph.reducer((left, right) => addMessages(left, right)),
  email_input: z.any(),
  classification_decision: z
    .enum(["ignore", "respond", "notify"])
    .nullable()
    .default(null),
});

export const EmailAgentHITLState = z.object({
  messages: z
    .array(z.any()) // Using any to support all Message types
    .default(() => [])
    .langgraph.reducer((left, right) => [...left, ...right], z.array(z.any())),
  email_input: z.any(),
  classification_decision: z
    .enum(["ignore", "respond", "notify", "error"])
    .nullable()
    .default(null),
});

// Export the inferred types from the Zod schemas
export type BaseEmailAgentStateType = z.infer<typeof BaseEmailAgentState>;
export type EmailAgentHITLStateType = z.infer<typeof EmailAgentHITLState>;

/**
 * Router schema for analyzing unread emails and routing based on content
 */
export const RouterSchema = z.object({
  reasoning: z
    .string()
    .describe("Step-by-step reasoning behind the classification"),
  classification: z
    .enum(["ignore", "respond", "notify"])
    .describe(
      "The classification of an email: 'ignore' for irrelevant emails, " +
      "'notify' for important information that doesn't need a response, " +
      "'respond' for emails that need a reply",
    ),
});

export type RouterOutput = z.infer<typeof RouterSchema>;

/**
 * Email data structure
 */
export type EmailData = {
  id: string;
  thread_id: string;
  from_email: string;
  subject: string;
  page_content: string;
  send_time: string;
  to_email: string;
};

/**
 * Input to the state
 */
export type StateInput = {
  email_input: EmailData;
};

/**
 * Core state definition for the graph
 */
export type State = {
  messages: BaseMessage[];
  email_input: EmailData;
  classification_decision?: "ignore" | "respond" | "notify";
};
