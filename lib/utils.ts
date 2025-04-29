import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Email Assistant utilities
 */

// Types
interface Message {
  role?: string;
  content: any;
  tool_calls?: ToolCall[];
  toolCalls?: ToolCall[];
  pretty_print?: () => void;
}

interface ToolCall {
  name: string;
  args: Record<string, any>;
}

interface Example {
  value: string;
}

/**
 * Format email details into a nicely formatted markdown string for display
 */
export function formatEmailMarkdown(
  subject: string, 
  author: string, 
  to: string, 
  emailThread: string
): string {
  return `
**Subject**: ${subject}
**From**: ${author}
**To**: ${to}

${emailThread}

---
`;
}

/**
 * Format content for display in Agent Inbox
 * @param state Current message state
 * @param toolCall The tool call to format
 */
export function formatForDisplay(
  state: { messages: Message[] }, 
  toolCall: ToolCall
): string {
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
**Attendees**: ${toolCall.args.attendees?.join(', ')}
**Duration**: ${toolCall.args.duration_minutes} minutes
**Day**: ${toolCall.args.preferred_day}
`;
      break;
    
    case "Question":
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
 * Parse an email input dictionary, supporting multiple schemas.
 * 
 * Supports multiple schema formats:
 * - Standard schema (author, to, subject, email_thread)
 * - Gmail-specific schema (from_email, to_email, subject, page_content)
 * - Direct API schema (from, to, subject, body)
 */
export function parseEmail(emailInput: Record<string, any>): [string, string, string, string] {
  // Detect schema based on keys present in the input
  if ("author" in emailInput && "email_thread" in emailInput) {
    // Standard schema
    return [
      emailInput.author,
      emailInput.to,
      emailInput.subject,
      emailInput.email_thread,
    ];
  } else if ("from_email" in emailInput && "page_content" in emailInput) {
    // Gmail schema
    return [
      emailInput.from_email,
      emailInput.to_email,
      emailInput.subject,
      emailInput.page_content,
    ];
  } else if ("from" in emailInput && "body" in emailInput) {
    // Direct API schema
    return [
      emailInput.from,
      emailInput.to,
      emailInput.subject,
      emailInput.body,
    ];
  } else {
    // Unknown schema, try to handle gracefully by looking for equivalent fields
    const author = 
      emailInput.author || 
      emailInput.from_email || 
      emailInput.from || 
      "Unknown Sender";
      
    const to = 
      emailInput.to || 
      emailInput.to_email || 
      "Unknown Recipient";
      
    const subject = emailInput.subject || "No Subject";
    
    const content = 
      emailInput.email_thread || 
      emailInput.page_content || 
      emailInput.body || 
      emailInput.content || 
      "No content available";
      
    return [author, to, subject, content];
  }
}

/**
 * Extract content from different message types as clean string.
 */
export function extractMessageContent(message: any): string {
  const content = message.content;
  
  // Check for recursion marker in string
  if (typeof content === 'string' && content.includes('<Recursion on AIMessage with id=')) {
    return "[Recursive content]";
  }
  
  // Handle string content
  if (typeof content === 'string') {
    return content;
  }
    
  // Handle list content (AIMessage format)
  else if (Array.isArray(content)) {
    const textParts: string[] = [];
    for (const item of content) {
      if (typeof item === 'object' && item !== null && 'text' in item) {
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
    const parts = example.value.split('Original routing:');
    const emailPart = parts[0].trim();
    
    const routingParts = parts[1].split('Correct routing:');
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
  return typeof item === 'object' && 
         item !== null && 
         'name' in item && 
         'args' in item;
}

/**
 * Extract tool call names from messages, safely handling messages without tool_calls.
 */
export function extractToolCalls(messages: any[]): string[] {
  const toolCallNames: string[] = [];
  
  for (const message of messages) {
    // Check if message is an object and has tool_calls
    if (typeof message === 'object' && message !== null) {
      // Handle plain objects
      if (message.tool_calls && Array.isArray(message.tool_calls)) {
        toolCallNames.push(...message.tool_calls.map(
          (call: any) => call.name.toLowerCase())
        );
      }
      // Handle class instances with toolCalls property
      else if ('toolCalls' in message && Array.isArray(message.toolCalls)) {
        toolCallNames.push(...message.toolCalls.map(
          (call: any) => call.name.toLowerCase())
        );
      }
    }
  }
  
  return toolCallNames;
}

/**
 * Format messages into a single string for analysis.
 * Note: This TypeScript implementation differs from the Python version
 * since we don't use stdout redirection.
 */
export function formatMessagesString(messages: Message[]): string {
  return messages.map(message => {
    let prefix = '';
    
    // Determine prefix based on role
    if ('role' in message && message.role) {
      switch (message.role) {
        case 'user':
          prefix = '🧑 Human: ';
          break;
        case 'assistant':
          prefix = '🤖 Assistant: ';
          break;
        case 'system':
          prefix = '🧠 System: ';
          break;
        case 'tool':
          prefix = '🛠️ Tool: ';
          break;
        default:
          prefix = `${message.role}: `;
      }
    }
    
    // Format content
    let content = typeof message.content === 'string' 
      ? message.content 
      : JSON.stringify(message.content);
    
    // Add tool calls if present
    const toolCalls = message.tool_calls || message.toolCalls;
    if (toolCalls && toolCalls.length > 0) {
      const toolCallsStr = toolCalls.map(tc => 
        `\n  Tool: ${tc.name}\n  Args: ${JSON.stringify(tc.args, null, 2)}`
      ).join('\n');
      
      content += `\n[Tool Calls: ${toolCallsStr}]`;
    }
    
    return `${prefix}${content}`;
  }).join('\n\n');
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
