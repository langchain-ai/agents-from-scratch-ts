/**
 * Email Assistant Tool Definitions
 *
 * This file contains the central registry for all tools available to the email assistant.
 * Tools are defined using the StructuredTool pattern from LangChain.
 */
import { writeEmail, triageEmail, Done } from "./default/email-tools.js";
import { scheduleMeeting, checkCalendarAvailability } from "./default/calendar-tools.js";
import { backgroundTool, calPreferencesTool, responsePreferencesTool, questionTool } from "./default/memory-tools.js";
/**
 * Returns an array of tools based on the provided options
 *
 * @param options - Configuration options for tool selection
 * @returns Array of StructuredTool instances ready for use with agents
 */
export async function getTools({ toolNames, includeGmail = false } = {}) {
    // Base tools dictionary - all available tools should be registered here
    const allTools = {
        write_email: writeEmail,
        triage_email: triageEmail,
        Done: Done,
        schedule_meeting: scheduleMeeting,
        check_calendar_availability: checkCalendarAvailability,
        background: backgroundTool,
        cal_preferences: calPreferencesTool,
        response_preferences: responsePreferencesTool,
        Question: questionTool,
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
export async function getToolsByName(tools) {
    const toolsList = tools || await getTools();
    return toolsList.reduce((acc, tool) => {
        acc[tool.name] = tool;
        return acc;
    }, {});
}
