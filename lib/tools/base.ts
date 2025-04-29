import { StructuredTool } from "@langchain/core/tools";
import { writeEmail, triageEmail, Done } from "./default/email-tools";
import { scheduleMeeting, checkCalendarAvailability } from "./default/calendar-tools";

interface GetToolsOptions {
  toolNames?: string[];
  includeGmail?: boolean;
}

export function getTools({ toolNames, includeGmail = false }: GetToolsOptions = {}): StructuredTool[] {
  // Base tools dictionary
  const allTools: Record<string, StructuredTool> = {
    write_email: writeEmail,
    triage_email: triageEmail,
    Done: Done,
    schedule_meeting: scheduleMeeting,
    check_calendar_availability: checkCalendarAvailability,
  };
  
  // Add Gmail tools if requested
  if (includeGmail) {
    try {
      // Import Gmail tools dynamically
      const gmailTools = require("./gmail/gmail-tools");
      
      Object.assign(allTools, {
        fetch_emails_tool: gmailTools.fetchEmails,
        send_email_tool: gmailTools.sendEmail,
        check_calendar_tool: gmailTools.checkCalendar,
        schedule_meeting_tool: gmailTools.scheduleMeeting,
      });
    } catch (error) {
      // If Gmail tools aren't available, continue without them
      console.warn("Gmail tools not available:", error);
    }
  }
  
  if (!toolNames) {
    return Object.values(allTools);
  }
  
  return toolNames
    .filter(name => name in allTools)
    .map(name => allTools[name]);
}

export function getToolsByName(tools?: StructuredTool[]): Record<string, StructuredTool> {
  const toolsList = tools || getTools();
  
  return toolsList.reduce<Record<string, StructuredTool>>((acc, tool) => {
    acc[tool.name] = tool;
    return acc;
  }, {});
} 