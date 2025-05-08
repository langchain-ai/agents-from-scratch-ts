/**
 * Memory-related tools for the Email Assistant with HITL and Memory
 * 
 * These tools allow the agent to retrieve information from the memory store.
 */

import { z } from "zod";
import { tool } from "@langchain/core/tools";

/**
 * Tool to retrieve background information about the user
 */
export const backgroundTool = tool(
  async () => {
    // This tool is a placeholder as the actual functionality happens in the agent implementation
    // It's used as a signal for the agent to include background information
    return "I'm Lance, a software engineer at LangChain.";
  },
  {
    name: "background",
    description: "Get background information about the user",
    schema: z.object({}).describe("This tool doesn't take any arguments")
  }
);

/**
 * Tool to retrieve calendar preferences
 */
export const calPreferencesTool = tool(
  async () => {
    // This tool is a placeholder as the actual functionality happens in the agent implementation
    // It's used as a signal for the agent to include calendar preferences
    return "30 minute meetings are preferred, but 15 minute meetings are also acceptable.";
  },
  {
    name: "cal_preferences",
    description: "Get the user's calendar preferences",
    schema: z.object({}).describe("This tool doesn't take any arguments")
  }
);

/**
 * Tool to retrieve response preferences
 */
export const responsePreferencesTool = tool(
  async () => {
    // This tool is a placeholder as the actual functionality happens in the agent implementation
    // It's used as a signal for the agent to include response preferences
    return "Use professional and concise language. If the e-mail mentions a deadline, make sure to explicitly acknowledge and reference the deadline in your response.";
  },
  {
    name: "response_preferences",
    description: "Get the user's response style preferences",
    schema: z.object({}).describe("This tool doesn't take any arguments")
  }
);

/**
 * Tool to ask a question to the user
 */
export const questionTool = tool(
  async ({ content }: { content: string }) => {
    return `The user will see and can answer this question: ${content}`;
  },
  {
    name: "Question",
    description: "Ask the user a follow-up question",
    schema: z.object({
      content: z.string().describe("The question to ask the user")
    })
  }
); 