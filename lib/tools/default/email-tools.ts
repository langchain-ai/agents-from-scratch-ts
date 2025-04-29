import { z } from "zod";
import { DynamicStructuredTool } from "@langchain/core/tools";

const emailSchema = z.object({
  recipient: z.string().describe("Email recipient"),
  subject: z.string().describe("Email subject"),
  content: z.string().describe("Email content")
});

export const writeEmail = new DynamicStructuredTool({
  name: "write_email",
  description: "Write an email draft based on provided information",
  schema: emailSchema,
  func: async ({ recipient, subject, content }: z.infer<typeof emailSchema>) => {
    // Mock implementation
    return `Email draft created to: ${recipient}\nSubject: ${subject}\n\n${content}`;
  }
});

const triageSchema = z.object({
  sender: z.string().describe("Email sender"),
  subject: z.string().describe("Email subject"),
  content: z.string().describe("Email content")
});

export const triageEmail = new DynamicStructuredTool({
  name: "triage_email",
  description: "Categorize and prioritize an email",
  schema: triageSchema,
  func: async ({ sender, subject, content }: z.infer<typeof triageSchema>) => {
    // Mock implementation
    return `Email from ${sender} triaged. Priority: Medium, Category: General`;
  }
});

const doneSchema = z.object({});

export const Done = new DynamicStructuredTool({
  name: "Done",
  description: "Mark the current task as complete",
  schema: doneSchema,
  func: async () => {
    return "Task completed successfully";
  }
}); 