import { EmailData } from "./schemas.js";

import {
  AIMessage,
  isAIMessage,
  coerceMessageLikeToMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import type { ToolCall } from "@langchain/core/messages/tool";

/**
 * Email Assistant utilities
 */

// Types
interface Example {
  value: string;
}

/**
 * Email parsing and formatting utilities
 */

/**
 * Parses an email to extract key information
 * @param email The email data to parse
 * @returns An object with { author, to, subject, emailThread }
 */
export function parseEmail(email: EmailData): {
  author: string;
  to: string;
  subject: string;
  emailThread: string;
} {
  try {
    // Extract key information from email data
    const author = email.from_email;
    const to = email.to_email;
    const subject = email.subject;
    const emailThread = email.page_content;

    return { author, to, subject, emailThread };
  } catch (error) {
    console.error("Error parsing email:", error);
    throw new Error("Failed to parse email");
  }
}

/**
 * Formats email data into a standardized markdown format for LLM processing
 * @param subject Email subject
 * @param author Email sender
 * @param to Email recipient
 * @param emailThread Email body content
 * @returns Formatted markdown string
 */
export function formatEmailMarkdown(
  subject: string,
  author: string,
  to: string,
  emailThread: string
): string {
  return `## Email: ${subject}

**From**: ${author}
**To**: ${to}

${emailThread}`;
}

/**
 * Format content for display in Agent Inbox
 * @param state Current message state
 * @param toolCall The tool call to format
 */
export function formatForDisplay(toolCall: ToolCall): string {
  // Initialize empty display
  let display = "";

  // Add tool call information based on tool type
  switch (toolCall.name) {
    case "write_email":
      display += `# Email Draft

**To**: ${toolCall.args.to}
**Subject**: ${toolCall.args.subject}

${toolCall.args.content}
`;
      break;

    case "schedule_meeting":
      display += `# Calendar Invite

**Meeting**: ${toolCall.args.subject}
**Attendees**: ${toolCall.args.attendees?.join(", ")}
**Duration**: ${toolCall.args.duration_minutes} minutes
**Day**: ${toolCall.args.preferred_day}
`;
      break;

    case "question":
      // Special formatting for questions to make them clear
      display += `# Question for User

${toolCall.args.content}
`;
      break;

    default:
      // Generic format for other tools
      display += `# Tool Call: ${toolCall.name}

Arguments:
${JSON.stringify(toolCall.args, null, 2)}
`;
  }

  return display;
}

/**
 * Extract content from different message types as clean string.
 */
export function extractMessageContent(message: any): string {
  const content = message.content;

  // Check for recursion marker in string
  if (
    typeof content === "string" &&
    content.includes("<Recursion on AIMessage with id=")
  ) {
    return "[Recursive content]";
  }

  // Handle string content
  if (typeof content === "string") {
    return content;
  }

  // Handle list content (AIMessage format)
  else if (Array.isArray(content)) {
    const textParts: string[] = [];
    for (const item of content) {
      if (typeof item === "object" && item !== null && "text" in item) {
        textParts.push(item.text);
      }
    }
    return textParts.join("\n");
  }

  // Don't try to handle recursion to avoid infinite loops
  // Just return string representation instead
  return String(content);
}

/**
 * Format examples into a readable string representation.
 */
export function formatFewShotExamples(examples: Example[]): string {
  const formatted: string[] = [];

  for (const example of examples) {
    // Parse the example value string into components
    const parts = example.value.split("Original routing:");
    const emailPart = parts[0].trim();

    const routingParts = parts[1].split("Correct routing:");
    const originalRouting = routingParts[0].trim();
    const correctRouting = routingParts[1].trim();

    // Format into clean string
    const formattedExample = `Example:
Email: ${emailPart}
Original Classification: ${originalRouting}
Correct Classification: ${correctRouting}
---`;

    formatted.push(formattedExample);
  }

  return formatted.join("\n");
}

/**
 * Type guard for checking if an object is a ToolCall
 */
export function isToolCall(item: any): item is ToolCall {
  return (
    typeof item === "object" &&
    item !== null &&
    "name" in item &&
    "args" in item
  );
}

/**
 * Extract tool call names from messages, safely handling messages without tool_calls.
 */
export function extractToolCalls(messages: any[]): string[] {
  const toolCallNames: string[] = [];

  for (const message of messages) {
    // Check if message is an object and has tool_calls
    if (typeof message === "object" && message !== null) {
      // Handle plain objects
      if (message.tool_calls && Array.isArray(message.tool_calls)) {
        toolCallNames.push(
          ...message.tool_calls.map((call: any) => call.name.toLowerCase())
        );
      }
      // Handle class instances with toolCalls property
      else if ("toolCalls" in message && Array.isArray(message.toolCalls)) {
        toolCallNames.push(
          ...message.toolCalls.map((call: any) => call.name.toLowerCase())
        );
      }
    }
  }

  return toolCallNames;
}

/**
 * Format messages into a single string for analysis.

 */
export function formatMessagesString(messages: BaseMessage[]): string {
  return messages
    .map((messageLike) => {
      let prefix = "";
      const message = coerceMessageLikeToMessage(messageLike);

      // Determine prefix based on role
      if ("role" in message && message.role) {
        switch (message.role) {
          case "user":
            prefix = "🧑 Human: ";
            break;
          case "assistant":
            prefix = "🤖 Assistant: ";
            break;
          case "system":
            prefix = "🧠 System: ";
            break;
          case "tool":
            prefix = "🛠️ Tool: ";
            break;
          default:
            prefix = `${message.role}: `;
        }
      }

      // Format content
      let content =
        typeof message.content === "string"
          ? message.content
          : JSON.stringify(message.content);

      // Only AIMessage can have tool_calls
      if (isAIMessage(message)) {
        const aiMessage = message as AIMessage; // Cast to AIMessage from @langchain/core/messages
        if (aiMessage.tool_calls && aiMessage.tool_calls.length > 0) {
          const toolCallsStr = aiMessage.tool_calls
            .map(
              (tc: ToolCall) =>
                `\n  Tool: ${tc.name}\n  Args: ${JSON.stringify(tc.args, null, 2)}`
            )
            .join("\n");
          content += `\n[Tool Calls: ${toolCallsStr}]`;
        }
      }

      return `${prefix}${content}`;
    })
    .join("\n\n");
}

/**
 * Format email with optional parameters
 */
export function formatEmailOptional(
  subject?: string,
  author?: string,
  to?: string,
  emailThread?: string
): string {
  return formatEmailMarkdown(
    subject ?? "No Subject",
    author ?? "Unknown Sender",
    to ?? "Unknown Recipient",
    emailThread ?? ""
  );
}
