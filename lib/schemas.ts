import { z } from "zod";
import { BaseMessage } from "@langchain/core/messages";
import "@langchain/langgraph/zod";
import { StateGraph } from "@langchain/langgraph";



export const AgentStateSchema = z.object({
  messages: z
    .array(z.string())
    .default(() => [])
    .langgraph.reducer(
      (a, b) => a.concat(Array.isArray(b) ? b : [b]),
      z.union([z.string(), z.array(z.string())])
    ),
  question: z.string(),
  answer: z.string().min(1),
});



/**
 * Router schema for analyzing unread emails and routing based on content
 */
export const RouterSchema = z.object({
  reasoning: z.string().describe("Step-by-step reasoning behind the classification"),
  classification: z.enum(["ignore", "respond", "notify"]).describe(
    "The classification of an email: 'ignore' for irrelevant emails, " +
    "'notify' for important information that doesn't need a response, " +
    "'respond' for emails that need a reply"
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