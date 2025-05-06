/**
 * @fileoverview Basic Email Assistant
 * 
 * This script implements a basic email assistant that can triage incoming emails
 * and generate responses without human intervention.
 *
 * @module email_assistant
 * 
 * @structure
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                           Email Assistant                                │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │ COMPONENTS                                                               │
 * │ - LLM                   : GPT-4 model for decision making                │
 * │ - Tools                 : Collection of tools for agent actions          │
 * │                                                                          │
 * │ GRAPH NODES                                                              │
 * │ - triage_router         : Classifies emails (ignore/respond/notify)      │
 * │ - response_agent        : Subgraph for handling email responses          │
 * │   ├─ llm_call           : Generates responses or tool calls              │
 * │   └─ environment        : Tool execution node                            │
 * │                                                                          │
 * │ EDGES                                                                    │
 * │ - START → triage_router                                                  │
 * │ - triage_router → response_agent/END                                     │
 * │ - response_agent.START → llm_call                                        │
 * │ - llm_call → environment/END                                             │
 * │ - environment → llm_call                                                 │
 * │                                                                          │
 * │ KEY FUNCTIONS                                                            │
 * │ - initializeEmailAssistant(): Creates and configures the agent graph     │
 * │ - llmCallNode()         : Generates responses using LLM                  │
 * │ - triageRouterNode()    : Classifies incoming emails                     │
 * │ - shouldContinue()      : Routes graph based on agent output             │
 * └─────────────────────────────────────────────────────────────────────────┘
 */

// LangChain imports for chat models
import { initChatModel } from "langchain/chat_models/universal";
import { BaseChatModel as ChatModel } from "@langchain/core/language_models/chat_models";

// LangGraph imports
import { StructuredTool } from "@langchain/core/tools";
import { 
  StateGraph, 
  START, 
  END,
  Command
} from "@langchain/langgraph";

import { ToolCall } from "@langchain/core/messages/tool";
import { isToolMessage } from "@langchain/core/messages/tool";
import { ToolNode } from "@langchain/langgraph/prebuilt";

// Zod imports
import { z } from "zod";
import "@langchain/langgraph/zod";

// LOCAL IMPORTS
import {
  getTools,
  getToolsByName
} from "../lib/tools/base.js";
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
  defaultTriageInstructions,
  AGENT_TOOLS_PROMPT
} from "../lib/prompts.js";
import {
  RouterSchema,
  RouterOutput,
  EmailData,
  StateInput,
  State,
  BaseEmailAgentState,
  BaseEmailAgentStateType
} from "../lib/schemas.js";
import {
  parseEmail,
  formatEmailMarkdown
} from "../lib/utils.js";

// Message Types from LangGraph SDK
import { HumanMessage, SystemMessage, ToolMessage, AIMessage, Message } from "@langchain/langgraph-sdk";


// Helper for type checking
const hasToolCalls = (message: Message): message is AIMessage & { tool_calls: ToolCall[] } => {
  return message.type === "ai" && 
    "tool_calls" in message && 
    Array.isArray(message.tool_calls);
};

// Define proper TypeScript types for our state
type AgentStateType = BaseEmailAgentStateType;
// Define node names as a union type for better type safety
type AgentNodes = typeof START | typeof END | "llm_call" | "environment" | "triage_router" | "response_agent";

/**
 * Initialize and export the email assistant
 */
export const initializeEmailAssistant = async () => {
  // Get tools
  const tools = await getTools();
  const toolsByName = await getToolsByName(tools);
  
  // Initialize the LLM
  const llm = await initChatModel("openai:gpt-4", { 
    temperature: 0.0,
    openAIApiKey: process.env.OPENAI_API_KEY 
  });
  
  // Initialize the LLM for tool use
  const llmWithTools = llm.bindTools(tools, { toolChoice: "required" });
  
  // Create the LLM call node
  const llmCallNode = async (state: AgentStateType) => {
    /**
     * LLM decides whether to call a tool or not
     * This is the main decision-making node that generates responses or tool calls
     */
    const messages = [...state.messages];
    const systemPromptContent = agentSystemPrompt
      .replace("{tools_prompt}", AGENT_TOOLS_PROMPT)
      .replace("{background}", defaultBackground)
      .replace("{response_preferences}", defaultResponsePreferences)
      .replace("{cal_preferences}", defaultCalPreferences);
    
    // Run the LLM with the messages
    const response = await llmWithTools.invoke([
      { type: "system", content: systemPromptContent },
      ...messages
    ]);
    
    // Use explicit casting as the response is compatible with Message in runtime
    return {
      messages: [response as unknown as Message]
    };
  };
  
  // Create the tool node
  const toolNode = new ToolNode(tools);
  
  // Conditional edge function for routing
  const shouldContinue = (state: AgentStateType) => {
    /**
     * Route to environment for tool execution, or end if Done tool called
     * Similar to the Python version's should_continue function
     */
    const messages = state.messages;
    if (!messages || messages.length === 0) return END;
    
    const lastMessage = messages[messages.length - 1];
    
    if (hasToolCalls(lastMessage) && lastMessage.tool_calls.length > 0) {
      // Check if any tool call is the "Done" tool
      if (lastMessage.tool_calls.some(toolCall => toolCall.name === "Done")) {
        return END;
      }
      return "environment";
    }
    
    return END;
  };
  
  // Create the triage router node
  const triageRouterNode = async (state: AgentStateType) => {
    /**
     * Analyze email content to decide if we should respond, notify, or ignore.
     * 
     * The triage step prevents the assistant from wasting time on:
     * - Marketing emails and spam
     * - Company-wide announcements
     * - Messages meant for other teams
     */
    try {
      const { email_input } = state;
      const parseResult = parseEmail(email_input);
      
      // Validate parsing result
      if (!Array.isArray(parseResult) || parseResult.length !== 4) {
        throw new Error("Invalid email parsing result");
      }
      
      const [author, to, subject, emailThread] = parseResult;
      
      const systemPrompt = triageSystemPrompt
        .replace("{background}", defaultBackground)
        .replace("{triage_instructions}", defaultTriageInstructions);
        
      const userPrompt = triageUserPrompt
        .replace("{author}", author)
        .replace("{to}", to)
        .replace("{subject}", subject)
        .replace("{email_thread}", emailThread);
      
      // Create email markdown for Agent Inbox in case of notification  
      const emailMarkdown = formatEmailMarkdown(subject, author, to, emailThread);
      
      // Add JSON format instruction to the system prompt
      const jsonSystemPrompt = `${systemPrompt}\n\nProvide your response in the following JSON format:
{
  "reasoning": "your step-by-step reasoning",
  "classification": "ignore" | "respond" | "notify"
}`;
      
      // Use the regular LLM instead of withStructuredOutput
      const response = await llm.invoke([
        { type: "system", content: jsonSystemPrompt },
        { type: "human", content: userPrompt }
      ]);
      
      // Parse the JSON response manually
      let classification: "ignore" | "respond" | "notify" = "ignore"; // Default
      
      try {
        // Extract JSON from the response content
        const responseText = response.content.toString();
        const parsedResponse = JSON.parse(responseText);
        
        if (parsedResponse.classification && 
            ["ignore", "respond", "notify"].includes(parsedResponse.classification)) {
          classification = parsedResponse.classification;
        }
      } catch (parseError) {
        console.error("Error parsing LLM response as JSON:", parseError);
        console.log("Raw response:", response.content.toString());
      }
      
      let goto = END;
      let update: Partial<AgentStateType> = {
        classification_decision: classification
      };
      
      if (classification === "respond") {
        console.log("📧 Classification: RESPOND - This email requires a response");
        goto = "response_agent";
        
        update.messages = [
          { type: "human", content: `Respond to the email: ${emailMarkdown}` }
        ];
      } else if (classification === "ignore") {
        console.log("🚫 Classification: IGNORE - This email can be safely ignored");
      } else if (classification === "notify") {
        console.log("🔔 Classification: NOTIFY - This email contains important information");
      } else {
        throw new Error(`Invalid classification: ${classification}`);
      }
      
      return new Command({
        goto,
        update
      });
    } catch (error) {
      console.error("Error in triage router:", error);
      // In case of error, we default to END without processing the email
      return new Command({
        goto: END,
        update: {
          classification_decision: "ignore",
          messages: [
            { type: "system", content: `Error processing email: ${error instanceof Error ? error.message : String(error)}` }
          ]
        }
      });
    }
  };
  
  // Build agent subgraph
  const agentBuilder = new StateGraph<typeof BaseEmailAgentState, 
    AgentStateType, 
    Partial<AgentStateType>,
    AgentNodes>(BaseEmailAgentState)
    .addNode("llm_call", llmCallNode)
    .addNode("environment", toolNode)
    .addEdge(START, "llm_call")
    .addConditionalEdges(
      "llm_call",
      shouldContinue,
      {
        "environment": "environment",
        [END]: END
      }
    )
    .addEdge("environment", "llm_call");
  
  // Compile the agent subgraph
  const agent = agentBuilder.compile();
  
  // Build overall workflow
  const emailAssistantGraph = new StateGraph<typeof BaseEmailAgentState, 
    AgentStateType, 
    Partial<AgentStateType>,
    AgentNodes>(BaseEmailAgentState)
    .addNode("triage_router", triageRouterNode, { ends: ["response_agent", END] })
    .addNode("response_agent", agent)
    .addEdge(START, "triage_router");
  
  // Compile and return the email assistant
  return emailAssistantGraph.compile();
};

// Initialize and export email assistant directly - replaces getEmailAssistant
export const emailAssistant = initializeEmailAssistant();
