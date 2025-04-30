/**
 * Email Assistant Tool Definitions
 * 
 * This file contains the central registry for all tools available to the email assistant.
 * Tools are defined using the StructuredTool pattern from LangChain.
 */

import { StructuredTool } from "@langchain/core/tools";
import { writeEmail, triageEmail, Done } from "./default/email-tools";
import { scheduleMeeting, checkCalendarAvailability } from "./default/calendar-tools";

/**
 * Options for customizing tool selection
 */
export interface GetToolsOptions {
  /** Optional list of specific tool names to include */
  toolNames?: string[];
  /** Whether to include Gmail-specific tools */
  includeGmail?: boolean;
}

/**
 * Returns an array of tools based on the provided options
 * 
 * @param options - Configuration options for tool selection
 * @returns Array of StructuredTool instances ready for use with agents
 */
export function getTools({ toolNames, includeGmail = false }: GetToolsOptions = {}): StructuredTool[] {
  // Base tools dictionary - all available tools should be registered here
  const allTools: Record<string, StructuredTool> = {
    write_email: writeEmail,
    triage_email: triageEmail,
    Done: Done,
    schedule_meeting: scheduleMeeting,
    check_calendar_availability: checkCalendarAvailability,
  };
  
  // If specific tool names are provided, filter to only those tools
  if (toolNames) {
    return toolNames
      .filter(name => name in allTools)
      .map(name => allTools[name]);
  }
  
  // Otherwise return all tools
  return Object.values(allTools);
}

/**
 * Creates a lookup map of tools by their name for easier access
 * 
 * @param tools - Optional array of tools to convert to lookup map
 * @returns Record mapping tool names to their corresponding StructuredTool instances
 */
export function getToolsByName(tools?: StructuredTool[]): Record<string, StructuredTool> {
  const toolsList = tools || getTools();
  
  return toolsList.reduce<Record<string, StructuredTool>>((acc, tool) => {
    acc[tool.name] = tool;
    return acc;
  }, {});
}