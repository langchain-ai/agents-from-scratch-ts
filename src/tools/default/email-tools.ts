import { z } from "zod";
import { tool } from "@langchain/core/tools";

/**
 * Schema for writing an email
 * Defines the required fields for composing a new email
 */
const emailSchema = z.object({
  recipient: z.string().describe("Email address of the recipient"),
  subject: z.string().describe("Clear and concise subject line for the email"),
  content: z.string().describe("Main body text of the email"),
});

/**
 * Tool for drafting emails
 * This tool allows the agent to compose email drafts based on user requests
 */
export const writeEmail = tool(
  async ({ recipient, subject, content }: z.infer<typeof emailSchema>) => {
    // In a real implementation, this would interact with an email service API
    // For now, we return a formatted string representation of the draft
    return `Email draft created:
To: ${recipient}
Subject: ${subject}

${content}

[Draft saved. Ready to send or edit further.]`;
  },
  {
    name: "write_email",
    description:
      "Write an email draft based on provided information. Use this when the user wants to compose a new email message.",
    schema: emailSchema,
  },
);

/**
 * Schema for triaging an email
 * Defines the required fields for analyzing and categorizing an email
 */
const triageSchema = z.object({
  sender: z.string().describe("Email address of the sender"),
  subject: z.string().describe("Subject line of the email to triage"),
  content: z.string().describe("Content/body of the email to analyze"),
});

/**
 * Tool for email triage
 * This tool helps categorize and prioritize incoming emails
 */
export const triageEmail = tool(
  async ({ sender, subject, content }: z.infer<typeof triageSchema>) => {
    // In a real implementation, this would use some classification logic
    // For demonstration, we return a mock categorization

    // Simple keyword-based priority assessment
    let priority = "Medium";
    if (
      subject.toLowerCase().includes("urgent") ||
      subject.toLowerCase().includes("important") ||
      content.toLowerCase().includes("asap")
    ) {
      priority = "High";
    } else if (content.length < 50 || subject.toLowerCase().includes("fyi")) {
      priority = "Low";
    }

    return `Email from ${sender} has been analyzed:
Priority: ${priority}
Category: General correspondence
Recommended action: ${priority === "High" ? "Respond immediately" : "Review when convenient"}`;
  },
  {
    name: "triage_email",
    description:
      "Analyze and categorize an email by importance and type. Use this when evaluating how to handle incoming messages.",
    schema: triageSchema,
  },
);

/**
 * Schema for the Done tool
 */
const doneSchema = z.object({
  content: z.string().describe("The content of the email"),
});

/**
 * Tool to mark a task as complete
 * This is a utility tool that signals the agent has completed its current task
 */
export const Done = tool(
  async () => {
    return "Task completed successfully. No further actions required.";
  },
  {
    name: "Done",
    description:
      "Signal that you've completed the current task and no further actions are needed.",
    schema: doneSchema,
  },
);
