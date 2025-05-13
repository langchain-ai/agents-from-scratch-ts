import { z } from "zod";
import { tool } from "@langchain/core/tools";

const scheduleMeetingSchema = z.object({
  title: z.string().describe("Meeting title"),
  attendees: z.array(z.string()).describe("List of attendees' emails"),
  startTime: z.string().describe("Meeting start time in ISO format"),
  endTime: z.string().describe("Meeting end time in ISO format"),
  description: z.string().optional().describe("Meeting description"),
});

export const scheduleMeeting = tool(
  async (args: z.infer<typeof scheduleMeetingSchema>) => {
    const { title, attendees, startTime, endTime, description } = args;
    // Mock implementation
    return `Meeting "${title}" scheduled from ${startTime} to ${endTime} with ${attendees.length} attendees`;
  },
  {
    name: "schedule_meeting",
    description: "Schedule a meeting on the calendar",
    schema: scheduleMeetingSchema,
  },
);

export const availabilitySchema = z.object({
  startTime: z.string().describe("Start time in ISO format"),
  endTime: z.string().describe("End time in ISO format"),
});

export const checkCalendarAvailability = tool(
  async (args: z.infer<typeof availabilitySchema>) => {
    const { startTime, endTime } = args;
    // Mock implementation
    return `Time slot from ${startTime} to ${endTime} is available`;
  },
  {
    name: "check_calendar_availability",
    description: "Check calendar availability for a specified time range",
    schema: availabilitySchema,
  },
);
