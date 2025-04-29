import Image from "next/image";
// LangChain imports for chat models

// Tool imports
// Note: Custom tools will need to be created in src/email_assistant/tools directory

// LangGraph imports
import { StructuredTool } from "@langchain/core/tools";
import { StateGraph, START, END } from "@langchain/langgraph";
import { Command } from "@langchain/langgraph";

// LOCAL IMPORTS
import {
  getTools,
  getToolsByName
} from "../lib/tools/base";
import {
  agentSystemPromptBaseline,
  triageSystemPrompt,
  triageUserPrompt,
  agentSystemPrompt,
  agentSystemPromptHitl,
  agentSystemPromptHitlMemory,
  defaultBackground,
  defaultResponsePreferences,
  defaultCalPreferences,
  defaultTriageInstructions
} from "../lib/prompts";
import {
  RouterSchema,
  RouterOutput,
  EmailData,
  StateInput,
  State,
} from "../lib/schemas";
import {
  parseEmail,
  formatEmailMarkdown
} from "../lib/utils";


export default function Home() {
  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <h1>Hello World</h1>
    </div>
  );
}
