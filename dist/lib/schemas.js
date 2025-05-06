import { z } from "zod";
import "@langchain/langgraph/zod";
// Define the Zod schemas for the email assistant states
export const BaseEmailAgentState = z.object({
    messages: z
        .array(z.any()) // Using any to support all Message types
        .default(() => [])
        .langgraph.reducer((left, right) => [...left, ...right], z.array(z.any())),
    email_input: z.any(),
    classification_decision: z.enum(["ignore", "respond", "notify"]).nullable().default(null)
});
export const EmailAgentHITLState = z.object({
    messages: z
        .array(z.any()) // Using any to support all Message types
        .default(() => [])
        .langgraph.reducer((left, right) => [...left, ...right], z.array(z.any())),
    email_input: z.any(),
    classification_decision: z.enum(["ignore", "respond", "notify", "error"]).nullable().default(null)
});
/**
 * Router schema for analyzing unread emails and routing based on content
 */
export const RouterSchema = z.object({
    reasoning: z.string().describe("Step-by-step reasoning behind the classification"),
    classification: z.enum(["ignore", "respond", "notify"]).describe("The classification of an email: 'ignore' for irrelevant emails, " +
        "'notify' for important information that doesn't need a response, " +
        "'respond' for emails that need a reply"),
});
