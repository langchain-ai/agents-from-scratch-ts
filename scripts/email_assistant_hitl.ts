// LangChain imports for chat models
import { initChatModel } from "langchain/chat_models/universal";

// LangGraph imports
import { 
  StateGraph, 
  START, 
  END,
  Command,
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
  HITL_TOOLS_PROMPT,
  triageSystemPrompt,
  triageUserPrompt,
  agentSystemPromptHitl,
  defaultBackground,
  defaultResponsePreferences,
  defaultCalPreferences,
  defaultTriageInstructions
} from "../lib/prompts.js";
import {
  RouterSchema,
  RouterOutput,
  EmailData,
  StateInput,
  State,
  EmailAgentHITLState,
  EmailAgentHITLStateType
} from "../lib/schemas.js";
import {
  parseEmail,
  formatEmailMarkdown,
  formatForDisplay
} from "../lib/utils.js";

// Message Types from LangGraph SDK
import { 
  HumanMessage, 
  SystemMessage, 
  ToolMessage, 
  AIMessage, 
  Message 
} from "@langchain/langgraph-sdk";

// Create message factory functions for types
const createSystemMessage = (content: string): SystemMessage => {
  return { type: "system", content, additional_kwargs: {} } as SystemMessage;
};

const createHumanMessage = (content: string): HumanMessage => {
  return { type: "human", content, additional_kwargs: {} } as HumanMessage;
};

const createToolMessage = (content: string, tool_call_id: string): ToolMessage => {
  return { type: "tool", content, tool_call_id, additional_kwargs: {} } as ToolMessage;
};

const createAIMessage = (content: string, tool_calls?: ToolCall[]): AIMessage => {
  return { 
    type: "ai", 
    content, 
    tool_calls: tool_calls || [],
    additional_kwargs: {} 
  } as AIMessage;
};

// Helper for type checking
const hasToolCalls = (message: Message): message is AIMessage & { tool_calls: ToolCall[] } => {
  return message.type === "ai" && 
    (message as any).tool_calls !== undefined && 
    Array.isArray((message as any).tool_calls);
};

/**
 * Create the Human-in-the-Loop Email Assistant
 * Mirrors the Python implementation in email_assistant_hitl.py
 */
export const createHitlEmailAssistant = async () => {
  // Get tools
  const tools = await getTools();
  const toolsByName = await getToolsByName();
  
  // Initialize the LLM
  const llm = await initChatModel("openai:gpt-4");
  
  // Initialize the LLM instance for tool use
  const llmWithTools = llm.bindTools(tools, { toolChoice: "required" });
  
  // Use the Zod schema imported from schemas.ts
  const AgentState = EmailAgentHITLState;

  // Define proper TypeScript types for our state
  type AgentStateType = EmailAgentHITLStateType;
  
  /**
   * Main LLM call handling node
   */
  const llmCall = async (state: AgentStateType) => {
    const { messages } = state;
    
    // Set up system prompt for the agent
    const systemPrompt = agentSystemPromptHitl 
      .replace("{tools_prompt}", HITL_TOOLS_PROMPT)
      .replace("{background}", defaultBackground)
      .replace("{response_preferences}", defaultResponsePreferences)
      .replace("{calendar_preferences}", defaultCalPreferences);
      
    // Create full message history for the agent
    const allMessages = [
      createSystemMessage(systemPrompt),
      ...messages
    ];
    
    // Run the LLM with the messages
    const result = await llmWithTools.invoke(allMessages);
    
    // Return the AIMessage result
    return {
      messages: [result]
    };
  };
  
  /**
   * Handles human review of tool calls
   */
  const interruptHandler = async (state: AgentStateType): Promise<Command> => {
    // Store messages to be returned
    const result: Message[] = [];
    
    // Default goto is llm_call
    let goto: typeof END | "llm_call" = "llm_call";
    
    // Get the last message
    const lastMessage = state.messages[state.messages.length - 1];
    
    // Exit early if there are no tool calls
    if (!hasToolCalls(lastMessage) || !lastMessage.tool_calls || lastMessage.tool_calls.length === 0) {
      return new Command({
        goto,
        update: { messages: result }
      });
    }
    
    // Keep track of processed tool calls to ensure all get responses
    const processedToolCallIds = new Set<string>();
    
    // Handle only one tool call at a time for better human-in-the-loop experience
    let processedOneToolCall = false;
    
    // Iterate over the tool calls in the last message
    for (const toolCall of lastMessage.tool_calls) {
      // Skip if we've already processed one tool call to allow proper resuming
      if (processedOneToolCall) {
        continue;
      }
      
      // Get or create a valid tool call ID
      const callId = toolCall.id ?? `fallback-id-${Date.now()}`;
      
      // Allowed tools for HITL
      const hitlTools = ["write_email", "schedule_meeting", "Question"];
      
      // If tool is not in our HITL list, execute it directly without interruption
      if (!hitlTools.includes(toolCall.name)) {
        const tool = toolsByName[toolCall.name];
        if (!tool) {
          console.error(`Tool ${toolCall.name} not found`);
          result.push(createToolMessage(`Error: Tool ${toolCall.name} not found`, callId));
          processedToolCallIds.add(callId);
          processedOneToolCall = true;
          continue;
        }
        
        try {
          // Parse the args properly - if it's a string, parse it as JSON
          const parsedArgs = typeof toolCall.args === 'string' 
            ? JSON.parse(toolCall.args) 
            : toolCall.args;
            
          // Invoke the tool with properly formatted arguments
          const observation = await tool.invoke(parsedArgs);
          result.push(createToolMessage(observation, callId));
          processedToolCallIds.add(callId);
          processedOneToolCall = true;
        } catch (error: any) {
          console.error(`Error executing tool ${toolCall.name}:`, error);
          result.push(createToolMessage(`Error executing tool: ${error.message}`, callId));
          processedToolCallIds.add(callId);
          processedOneToolCall = true;
        }
        continue;
      }
      
      // Get original email from email_input in state
      const emailInput = state.email_input;
      const parseResult = parseEmail(emailInput);
      // Validate parsing result
      if (!Array.isArray(parseResult) || parseResult.length !== 4) {
        throw new Error("Invalid email parsing result");
      }
      const [author, to, subject, emailThread] = parseResult;
      const originalEmailMarkdown = formatEmailMarkdown(subject, author, to, emailThread);
      
      // Format tool call for display
      const toolDisplay = formatForDisplay(state, toolCall);
      const description = originalEmailMarkdown + toolDisplay;
      
      try {
        // Use the interrupt function from LangGraph
        const { interrupt } = await import("@langchain/langgraph");
        
        // IMPORTANT: We're directly passing the interrupt call result without modifying it
        const humanReview = await interrupt({
          question: "Review this tool call before execution:",
          toolCall: toolCall,
          description
        });
        
        const reviewAction = humanReview.action;
        const reviewData = humanReview.data;
        
        if (reviewAction === "continue") {
          // Execute the tool with original args
          const tool = toolsByName[toolCall.name];
          // Parse the args properly
          const parsedArgs = typeof toolCall.args === 'string' 
            ? JSON.parse(toolCall.args) 
            : toolCall.args;
            
          const observation = await tool.invoke(parsedArgs);
          result.push(createToolMessage(observation, callId));
          processedToolCallIds.add(callId);
          processedOneToolCall = true;
        } 
        else if (reviewAction === "update") {
          // Execute with edited args
          const tool = toolsByName[toolCall.name];
          // Make sure the updated args are properly formatted
          const updatedArgs = typeof reviewData === 'string' 
            ? JSON.parse(reviewData) 
            : reviewData;
            
          const observation = await tool.invoke(updatedArgs);
          result.push(createToolMessage(observation, callId));
          processedToolCallIds.add(callId);
          processedOneToolCall = true;
        }
        else if (reviewAction === "feedback") {
          // Add feedback as a tool message
          result.push(createToolMessage(reviewData, callId));
          processedToolCallIds.add(callId);
          processedOneToolCall = true;
          goto = "llm_call";
        }
        else if (reviewAction === "stop") {
          // Even when stopping, we still need to respond to the tool call
          result.push(createToolMessage("User chose to stop this action.", callId));
          processedToolCallIds.add(callId);
          processedOneToolCall = true;
          goto = END;
        }
        else {
          // Handle any other action by providing a default response
          result.push(createToolMessage("Action not recognized or canceled by user.", callId));
          processedToolCallIds.add(callId);
          processedOneToolCall = true;
          goto = END;
        }
      } catch (error: any) {
        // Very important: Just rethrow any GraphInterrupt error without modifying it
        // This ensures LangGraph can properly handle the interruption
        if (error.name === 'GraphInterrupt' || 
            (error.message && typeof error.message === 'string' && 
             error.message.includes('GraphInterrupt'))) {
          throw error;
        }
        
        console.error("Error with interrupt handler:", error);
        // For other errors, provide a message response
        result.push(createToolMessage(`Error during tool execution: ${error.message}`, callId));
        processedToolCallIds.add(callId);
        processedOneToolCall = true;
      }
    }
    
    // If we've processed a tool call, return right away
    if (processedOneToolCall) {
      return new Command({
        goto,
        update: { messages: result }
      });
    }
    
    // If we reach here and haven't processed any tool calls,
    // we need to return appropriate responses for any remaining ones
    lastMessage.tool_calls?.forEach(toolCall => {
      const callId = toolCall.id ?? `fallback-id-${Date.now()}`;
      if (!processedToolCallIds.has(callId)) {
        // We've skipped this tool call, but we still need to respond to it
        // This is important for OpenAI's API requirement that every tool call has a response
        result.push(createToolMessage("Tool execution pending human review.", callId));
      }
    });
    
    // Return the Command with goto and update
    return new Command({
      goto,
      update: { messages: result }
    });
  };
  
  /**
   * Analyzes email content to decide if we should respond, notify, or ignore
   */
  const triageRouter = async (state: AgentStateType) => {
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
        createSystemMessage(jsonSystemPrompt),
        createHumanMessage(userPrompt)
      ]);
      
      // Parse the JSON response manually
      let classification: "ignore" | "respond" | "notify" = "notify"; // Default to notify
      
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
        // Fall back to notify if parsing fails
      }
      
      let goto: "triage_interrupt_handler" | "response_agent" | typeof END = END;
      let update: Partial<AgentStateType> = {
        classification_decision: classification
      };
      
      // Create message
      update.messages = [
        createHumanMessage(`Email to review: ${emailMarkdown}`)
      ];
      
      if (classification === "respond") {
        console.log("📧 Classification: RESPOND - This email requires a response");
        goto = "response_agent";
      } else if (classification === "notify") {
        console.log("🔔 Classification: NOTIFY - This email contains important information");
        goto = "triage_interrupt_handler";
      } else if (classification === "ignore") {
        console.log("🚫 Classification: IGNORE - This email can be safely ignored");
        goto = END;
      } else {
        // Default to notify if classification is not recognized
        goto = "triage_interrupt_handler";
        update.classification_decision = "notify";
      }
      
      return new Command({
        goto,
        update
      });
    } catch (error: any) {
      console.error("Error in triage router:", error);
      return new Command({
        goto: END,
        update: {
          classification_decision: "error",
          messages: [
            createSystemMessage(`Error in triage router: ${error.message}`)
          ]
        }
      });
    }
  };
  
  /**
   * Handles interrupts from the triage step
   */
  const triageInterruptHandler = async (state: AgentStateType) => {
    // Parse the email input
    const parseResult = parseEmail(state.email_input);
    
    // Validate parsing result
    if (!Array.isArray(parseResult) || parseResult.length !== 4) {
      throw new Error("Invalid email parsing result");
    }
    
    const [author, to, subject, emailThread] = parseResult;
    
    // Create email markdown for Agent Inbox in case of notification  
    const emailMarkdown = formatEmailMarkdown(subject, author, to, emailThread);
    
    try {
      // Use the interrupt function from LangGraph
      const { interrupt } = await import("@langchain/langgraph");
      
      const humanReview = await interrupt({
        question: `Email requires attention: ${state.classification_decision || "notify"}`,
        email: emailMarkdown
      });
      
      let goto: "response_agent" | typeof END = END;
      const messages = [
        createHumanMessage(`Email to review: ${emailMarkdown}`)
      ];
      
      // Handle different response types
      const reviewAction = humanReview.action;
      const reviewData = humanReview.data;
      
      if (reviewAction === "continue") {
        // Human wants to handle this email - proceed to response agent
        goto = "response_agent";
      } else if (reviewAction === "feedback") {
        // Human provided feedback or instructions on how to handle
        messages.push(createHumanMessage(reviewData));
        goto = "response_agent";
      } else {
        // Default to END for other actions
        goto = END;
      }
      
      // Return Command with goto and update
      return new Command({
        goto,
        update: {
          messages
        }
      });
    } catch (error) {
      console.error("Error with triage interrupt handler:", error);
      return new Command({
        goto: END,
        update: {
          messages: [
            createSystemMessage(`Error in triage interrupt: ${error}`)
          ]
        }
      });
    }
  };
  
  /**
   * Route to tool handler, or end if Done tool called
   */
  const shouldContinue = (state: AgentStateType) => {
    /**
     * Similar to the Python version's should_continue function
     */
    const messages = state.messages;
    if (!messages || messages.length === 0) return END;
    
    const lastMessage = messages[messages.length - 1];
    
    if (hasToolCalls(lastMessage) && lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
      // Check if any tool call is the "Done" tool
      if (lastMessage.tool_calls.some((toolCall: ToolCall) => toolCall.name === "Done")) {
        return END;
      }
      return "interrupt_handler";
    }
    
    return END;
  };
  
  // Define node names as a union type for better type safety
  type AgentNodes = typeof START | typeof END | "llm_call" | "interrupt_handler" | "triage_router" | "triage_interrupt_handler" | "response_agent";

  // Build agent workflow with the builder pattern
  const agentBuilder = new StateGraph<typeof AgentState, 
    AgentStateType, 
    Partial<AgentStateType>,
    AgentNodes>(AgentState)
    .addNode("llm_call", llmCall)
    .addNode("interrupt_handler", interruptHandler);
  
  // Add edges
  agentBuilder.addEdge(START, "llm_call");
  agentBuilder.addConditionalEdges(
    "llm_call",
    shouldContinue,
    {
      "interrupt_handler": "interrupt_handler",
      [END]: END
    }
  );
  agentBuilder.addEdge("interrupt_handler", "llm_call");
  
  // Compile the agent
  const responseAgent = agentBuilder.compile();
  
  // Build overall workflow with the builder pattern
  const overallWorkflow = new StateGraph<typeof AgentState, 
    AgentStateType, 
    Partial<AgentStateType>,
    AgentNodes>(AgentState)
    .addNode("triage_router", triageRouter, {
      ends: ["triage_interrupt_handler", "response_agent", END]
    })
    .addNode("triage_interrupt_handler", triageInterruptHandler, {
      ends: ["response_agent", END]
    })
    .addNode("response_agent", responseAgent, {
      ends: [END]
    });
  
  overallWorkflow.addEdge(START, "triage_router");
  overallWorkflow.addEdge("response_agent", END);
  
  // Compile the email assistant
  const hitlEmailAssistant = overallWorkflow.compile();
  
  return hitlEmailAssistant;
};

// For server-side usage
export async function getHitlEmailAssistant() {
  return await createHitlEmailAssistant();
} 