import { z } from "zod";
import { BaseMessage } from "@langchain/core/messages";
import "@langchain/langgraph/zod";
import { addMessages, Messages, StateGraph } from "@langchain/langgraph";

export const MessagesState = z.object({
  messages: z
    .custom<BaseMessage[]>() // Using any to support all Message types
    .default(() => [])
    .langgraph.reducer<Messages>((left, right) => addMessages(left, right)),
});

// Define the Zod schemas for the email assistant states
export const BaseEmailAgentState = MessagesState.extend({
  email_input: z.any(),
  classification_decision: z
    .enum(["ignore", "respond", "notify"])
    .nullable()
    .default(null),
});

export const EmailAgentHITLState = MessagesState.extend({
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
