// LangChain imports for chat models
import { initChatModel } from "langchain/chat_models/universal";
import { BaseChatModel as ChatModel } from "@langchain/core/language_models/chat_models";

// LangGraph imports
import { 
  StateGraph, 
  START, 
  END,
  Command,
  MemorySaver,
  InMemoryStore,
  BaseStore
} from "@langchain/langgraph";
import { ToolCall } from "@langchain/core/messages/tool";
import { isToolMessage } from "@langchain/core/messages/tool";
import { StructuredTool } from "@langchain/core/tools";

// Zod imports
import { z } from "zod";
import "@langchain/langgraph/zod";

// Message Types from LangGraph SDK
import { 
  HumanMessage, 
  SystemMessage, 
  ToolMessage, 
  AIMessage, 
  Message 
} from "@langchain/langgraph-sdk";

// LOCAL IMPORTS
import {
  getTools,
  getToolsByName
} from "../lib/tools/base.js";
import {
  HITL_MEMORY_TOOLS_PROMPT,
  triageSystemPrompt,
  triageUserPrompt,
  agentSystemPromptHitlMemory,
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

/**
 * @file Email Assistant with Human-in-the-Loop and Memory
 * @description Modular implementation of an email assistant with human review capability and memory
 * 
 * @module EmailAssistantHITLMemory
 * 
 * @exports
 * @function setupLLMAndTools - Initializes LLM and tools for the assistant
 * @function getMemory - Retrieves memory from the store with fallback
 * @function updateMemory - Updates memory in the store based on user feedback
 * 
 * @function createTriageRouterNode - Implements email triage logic with memory
 * @function createTriageInterruptHandlerNode - Handles human review for triage with memory updates
 * @function createLLMCallNode - Creates the LLM node with memory integration
 * @function createInterruptHandlerNode - Creates the interrupt handler with memory updates
 * 
 * @function createShouldContinueFunction - Provides conditional routing logic
 * @function buildAgentGraph - Constructs the agent state graph
 * @function buildOverallWorkflow - Creates the complete workflow graph with memory
 * @function createHitlEmailAssistantWithMemory - Main function to initialize the assistant
 * @function getHitlEmailAssistantWithMemory - Server-side utility function 
 */

// Helper for type checking
const hasToolCalls = (message: Message): message is AIMessage & { tool_calls: ToolCall[] } => {
  return message.type === "ai" && 
    "tool_calls" in message && 
    Array.isArray(message.tool_calls);
};

// Define UserPreferences interface for memory updates
interface UserPreferences {
  preferences: string;
  justification: string;
}

// Define proper TypeScript types for our state
type AgentStateType = EmailAgentHITLStateType;
// Define node names as a union type for better type safety
type AgentNodes = typeof START | typeof END | "llm_call" | "interrupt_handler" | "triage_router" | "triage_interrupt_handler" | "response_agent";

// Constants for memory update prompts
const MEMORY_UPDATE_INSTRUCTIONS = `
# Role and Objective
You are a memory profile manager for an email assistant agent that selectively updates user preferences based on feedback messages from human-in-the-loop interactions with the email assistant.

# Instructions
- NEVER overwrite the entire memory profile
- ONLY make targeted additions of new information
- ONLY update specific facts that are directly contradicted by feedback messages
- PRESERVE all other existing information in the profile
- Format the profile consistently with the original style
- Generate the profile as a string

# Reasoning Steps
1. Analyze the current memory profile structure and content
2. Review feedback messages from human-in-the-loop interactions
3. Extract relevant user preferences from these feedback messages
4. Compare new information against existing profile
5. Identify only specific facts to add or update
6. Preserve all other existing information
7. Output the complete updated profile

# Process current profile for {namespace}
<memory_profile>
{current_profile}
</memory_profile>

Think step by step about what specific feedback is being provided and what specific information should be added or updated in the profile while preserving everything else.`;

const MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT = `
Remember:
- NEVER overwrite the entire profile
- ONLY make targeted additions or changes based on explicit feedback
- PRESERVE all existing information not directly contradicted
- Output the complete updated profile as a string
`;

/**
 * Initialize LLM and tools
 */
export const setupLLMAndTools = async () => {
  // Get tools
  const tools = await getTools();
  const toolsByName = await getToolsByName();
  
  // Initialize the LLM for use with router / structured output
  const llm = await initChatModel("openai:gpt-4");
  
  // Initialize the LLM, enforcing tool use (of any available tools) for agent
  const llmWithTools = llm.bindTools(tools, { toolChoice: "required" });
  
  return { llm, llmWithTools, tools, toolsByName };
};

/**
 * Get memory from the store or initialize with default if it doesn't exist.
 * 
 * @param store LangGraph BaseStore instance to search for existing memory
 * @param namespace Array defining the memory namespace, e.g. ["email_assistant", "triage_preferences"]
 * @param defaultContent Default content to use if memory doesn't exist
 * @returns The content of the memory profile, either from existing memory or the default
 */
async function getMemory(
  store: BaseStore,
  namespace: string[],
  defaultContent: string = ""
): Promise<string> {
  try {
    // Search for existing memory with namespace and key
    const userPreferences = await store.get(namespace, "user_preferences");
    
    // If memory exists, return its content (the value)
    if (userPreferences) {
      return userPreferences.value as unknown as string;
    }
    
    // If memory doesn't exist, add it to the store with default content
    await store.put(namespace, "user_preferences", { value: defaultContent });
    return defaultContent;
  } catch (error) {
    console.error("Error getting memory:", error);
    // Return default content if there's an error
    await store.put(namespace, "user_preferences", { value: defaultContent });
    return defaultContent;
  }
}

/**
 * Update memory profile in the store based on user messages.
 * 
 * @param store LangGraph BaseStore instance to update memory
 * @param namespace Array defining the memory namespace
 * @param messages List of messages to update the memory with
 */
async function updateMemory(
  store: InMemoryStore,
  namespace: string[],
  messages: Message[]
): Promise<void> {
  try {
    // Get the existing memory
    const userPreferences = await store.get(namespace, "user_preferences");
    const currentProfile = userPreferences ? userPreferences.value as unknown as string : "";
    
    // Convert Message[] to format expected by the LLM
    const formattedMessages = messages.map(msg => ({
      role: msg.type === "human" ? "user" : 
            msg.type === "ai" ? "assistant" : 
            "system",
      content: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content)
    }));
    
    // Initialize chat model
    const llm = await initChatModel("openai:gpt-4");
    const llmWithStructuredOutput = llm.withStructuredOutput<UserPreferences>({
      schema: {
        type: "object", 
        properties: {
          preferences: { type: "string" },
          justification: { type: "string" }
        },
        required: ["preferences"]
      }
    });
    
    // Create system message with memory update instructions
    const systemMsg = { 
      type: "system" as const, 
      content: MEMORY_UPDATE_INSTRUCTIONS.replace(
        "{current_profile}", 
        currentProfile
      ).replace(
        "{namespace}",
        namespace.join("/")
      )
    };
    
    // Create user message instructing to update memory
    const userMsg = { 
      type: "human" as const, 
      content: "Think carefully and update the memory profile based upon these user messages:" 
    };
    
    // Convert the formatted messages to Message objects
    const messagesForLLM = formattedMessages.map(msg => {
      if (msg.role === "user") {
        return { type: "human" as const, content: msg.content };
      } else if (msg.role === "assistant") {
        return { type: "ai" as const, content: msg.content, tool_calls: [] };
      } else {
        return { type: "system" as const, content: msg.content };
      }
    });
    
    // Invoke the LLM with the messages
    const result = await llmWithStructuredOutput.invoke([
      systemMsg,
      userMsg,
      ...messagesForLLM
    ]);
    
    // Save the updated memory to the store
    await store.put(namespace, "user_preferences", { value: result.preferences });
  } catch (error) {
    console.error("Error updating memory:", error);
  }
}

/**
 * Create the triage router node with memory integration
 */
export const createTriageRouterNode = (llm: ChatModel, store: InMemoryStore) => {
  return async (state: AgentStateType): Promise<Command> => {
    try {
      const { email_input } = state;
      const parseResult = parseEmail(email_input);
      
      // Validate parsing result
      if (!Array.isArray(parseResult) || parseResult.length !== 4) {
        throw new Error("Invalid email parsing result");
      }
      
      const [author, to, subject, emailThread] = parseResult;
      
      // Get triage preferences from memory
      const triagePreferences = await getMemory(
        store, 
        ["email_assistant", "triage_preferences"], 
        defaultTriageInstructions
      );
      
      const systemPrompt = triageSystemPrompt
        .replace("{background}", defaultBackground)
        .replace("{triage_instructions}", triagePreferences);
        
      const userPrompt = triageUserPrompt
        .replace("{author}", author)
        .replace("{to}", to)
        .replace("{subject}", subject)
        .replace("{email_thread}", emailThread);
      
      // Create email markdown for Agent Inbox in case of notification  
      const emailMarkdown = formatEmailMarkdown(subject, author, to, emailThread);
      
      // Add a clear instruction for a simple string response rather than JSON
      const simplifiedSystemPrompt = `${systemPrompt}

After analyzing this email, determine if it should be:
1. "ignore" - Not important, no action needed
2. "respond" - Requires a response
3. "notify" - Contains important information but no response needed

Reply with ONLY ONE WORD: "ignore", "respond", or "notify".`;
      
      // Use the regular LLM with a simplified prompt
      const response = await llm.invoke([
        { type: "system", content: simplifiedSystemPrompt },
        { type: "human", content: userPrompt }
      ]);
      
      // Extract the classification from the simple response
      let classification: "ignore" | "respond" | "notify";
      const responseText = (response.content || "").toString().toLowerCase().trim();
      
      if (responseText.includes("respond")) {
        classification = "respond";
      } else if (responseText.includes("notify")) {
        classification = "notify";
      } else if (responseText.includes("ignore")) {
        classification = "ignore";
      } else {
        console.log(`Unrecognized classification: "${responseText}". Defaulting to notify.`);
        classification = "notify";
      }
      
      let goto: "triage_interrupt_handler" | "response_agent" | typeof END = END;
      let update: Partial<AgentStateType> = {
        classification_decision: classification
      };
      
      if (classification === "respond") {
        console.log("📧 Classification: RESPOND - This email requires a response");
        goto = "response_agent";
        update.messages = [
          { type: "human", content: `Respond to the email: ${emailMarkdown}` }
        ];
      } else if (classification === "notify") {
        console.log("🔔 Classification: NOTIFY - This email contains important information");
        goto = "triage_interrupt_handler";
      } else if (classification === "ignore") {
        console.log("🚫 Classification: IGNORE - This email can be safely ignored");
        goto = END;
      } else {
        // Default to "ignore" if classification is not recognized
        console.log("❓ Classification: UNKNOWN - Treating as IGNORE");
        goto = END;
        update.classification_decision = "ignore";
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
            { type: "system", content: `Error in triage router: ${error.message}` }
          ]
        }
      });
    }
  };
};

/**
 * Create the triage interrupt handler node with memory integration
 */
export const createTriageInterruptHandlerNode = (store: InMemoryStore) => {
  return async (state: AgentStateType): Promise<Command> => {
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
      // Create messages
      const messages = [
        { type: "human" as const, content: `Email to notify user about: ${emailMarkdown}` }
      ];
      
      // Create interrupt for Agent Inbox
      const { interrupt } = await import("@langchain/langgraph");
      
      const humanReview = await interrupt({
        question: `Email requires attention: ${state.classification_decision || "notify"}`,
        email: emailMarkdown,
        options: {
          allowIgnore: true,
          allowRespond: true,
          allowEdit: false,
          allowAccept: false
        }
      });
      
      // Initialize return values
      let goto: "response_agent" | typeof END = END;
      const returnMessages = [...messages];
      
      // Handle different response types
      const reviewAction = humanReview.action;
      const reviewData = humanReview.data;
      
      if (reviewAction === "continue" || reviewAction === "feedback") {
        // User wants to respond to the email
        if (reviewData) {
          returnMessages.push(
            { type: "human" as const, content: `User wants to reply to the email. Use this feedback to respond: ${reviewData}` }
          );
        }
        
        // Update memory with feedback
        const memoryUpdateMessages = [
          { type: "human" as const, content: "The user decided to respond to the email, so update the triage preferences to capture this." },
          ...returnMessages
        ];
        
        await updateMemory(
          store, 
          ["email_assistant", "triage_preferences"], 
          memoryUpdateMessages
        );
        
        goto = "response_agent";
      } else {
        // User ignored the email or other action
        const memoryUpdateMessages = [
          { type: "human" as const, content: "The user decided to ignore the email even though it was classified as notify. Update triage preferences to capture this." },
          ...returnMessages
        ];
        
        // Update memory with feedback
        await updateMemory(
          store, 
          ["email_assistant", "triage_preferences"], 
          memoryUpdateMessages
        );
        
        goto = END;
      }
      
      // Return Command with goto and update
      return new Command({
        goto,
        update: {
          messages: returnMessages
        }
      });
    } catch (error: any) {
      console.error("Error with triage interrupt handler:", error);
      return new Command({
        goto: END,
        update: {
          messages: [
            { type: "system" as const, content: `Error in triage interrupt: ${error.message}` }
          ]
        }
      });
    }
  };
};

/**
 * Create the LLM call node with memory integration
 */
export const createLLMCallNode = (llmWithTools: ChatModel, store: InMemoryStore) => {
  return async (state: AgentStateType) => {
    try {
      // Get memory preferences
      const calPreferences = await getMemory(
        store,
        ["email_assistant", "cal_preferences"],
        defaultCalPreferences
      );
      
      const responsePreferences = await getMemory(
        store,
        ["email_assistant", "response_preferences"],
        defaultResponsePreferences
      );
      
      // Set up system prompt for the agent
      const systemPrompt = agentSystemPromptHitlMemory
        .replace("{tools_prompt}", HITL_MEMORY_TOOLS_PROMPT)
        .replace("{background}", defaultBackground)
        .replace("{response_preferences}", responsePreferences)
        .replace("{cal_preferences}", calPreferences);
        
      // Create full message history for the agent
      const allMessages = [
        { type: "system" as const, content: systemPrompt },
        ...state.messages
      ];
      
      // Run the LLM with the messages
      const result = await llmWithTools.invoke(allMessages);
      
      // Return the AIMessage result
      return {
        messages: [result as unknown as Message]
      };
    } catch (error: any) {
      console.error("Error in LLM call:", error);
      return {
        messages: [
          { type: "ai" as const, content: `Error calling model: ${error.message}`, tool_calls: [] }
        ]
      };
    }
  };
};

/**
 * Create the interrupt handler node with memory updates
 */
export const createInterruptHandlerNode = (toolsByName: Record<string, StructuredTool>, store: InMemoryStore) => {
  return async (state: AgentStateType): Promise<Command> => {
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
    
    // Process one tool call at a time for better human-in-the-loop experience
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
          result.push({ type: "tool", content: `Error: Tool ${toolCall.name} not found`, tool_call_id: callId });
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
          result.push({ type: "tool", content: observation, tool_call_id: callId });
          processedToolCallIds.add(callId);
          processedOneToolCall = true;
        } catch (error: any) {
          console.error(`Error executing tool ${toolCall.name}:`, error);
          result.push({ type: "tool", content: `Error executing tool: ${error.message}`, tool_call_id: callId });
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
      
      // Format tool call for display and prepend the original email
      const toolDisplay = formatForDisplay(state, toolCall);
      const description = originalEmailMarkdown + toolDisplay;
      
      try {
        // Configure what actions are allowed in Agent Inbox
        const options = {
          allowIgnore: true,
          allowRespond: true,
          allowEdit: toolCall.name === "write_email" || toolCall.name === "schedule_meeting",
          allowAccept: toolCall.name === "write_email" || toolCall.name === "schedule_meeting"
        };
        
        // Use the interrupt function from LangGraph
        const { interrupt } = await import("@langchain/langgraph");
        
        const humanReview = await interrupt({
          question: `Review this ${toolCall.name} action:`,
          toolCall,
          description,
          options
        });
        
        const reviewAction = humanReview.action;
        const reviewData = humanReview.data;
        
        if (reviewAction === "accept") {
          // Execute the tool with original args
          const tool = toolsByName[toolCall.name];
          // Parse the args properly
          const parsedArgs = typeof toolCall.args === 'string' 
            ? JSON.parse(toolCall.args) 
            : toolCall.args;
            
          const observation = await tool.invoke(parsedArgs);
          result.push({
            type: "tool",
            content: observation,
            tool_call_id: callId
          });
          processedToolCallIds.add(callId);
          processedOneToolCall = true;
        } 
        else if (reviewAction === "edit") {
          // Tool selection 
          const tool = toolsByName[toolCall.name];
          
          // Get edited args from Agent Inbox
          const editedArgs = reviewData?.args || {};
          
          // Save the initial tool call for memory updates
          const initialToolCall = `${toolCall.name}: ${JSON.stringify(toolCall.args)}`;
          
          // Update the AI message's tool call with edited content (reference to the message in the state)
          if (hasToolCalls(lastMessage)) {
            // Replace the original tool call with the edited one
            lastMessage.tool_calls = lastMessage.tool_calls?.map(tc => {
              if (tc.id === callId) {
                return {
                  ...tc,
                  args: editedArgs
                };
              }
              return tc;
            });
          }
          
          // Execute the tool with edited args
          const observation = await tool.invoke(editedArgs);
          
          // Add the tool response
          result.push({
            type: "tool",
            content: observation,
            tool_call_id: callId
          });
          processedToolCallIds.add(callId);
          processedOneToolCall = true;
          
          // Update memory with user edits
          if (toolCall.name === "write_email") {
            await updateMemory(
              store,
              ["email_assistant", "response_preferences"],
              [
                {
                  type: "human",
                  content: `User edited the email response. Here is the initial email generated by the assistant: ${initialToolCall}. Here is the edited email: ${JSON.stringify(editedArgs)}. Follow all instructions above, and remember: ${MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.`
                }
              ]
            );
          } else if (toolCall.name === "schedule_meeting") {
            await updateMemory(
              store,
              ["email_assistant", "cal_preferences"],
              [
                {
                  type: "human",
                  content: `User edited the calendar invitation. Here is the initial calendar invitation generated by the assistant: ${initialToolCall}. Here is the edited calendar invitation: ${JSON.stringify(editedArgs)}. Follow all instructions above, and remember: ${MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.`
                }
              ]
            );
          }
        }
        else if (reviewAction === "ignore") {
          if (toolCall.name === "write_email") {
            // Don't execute the tool, and tell the agent how to proceed
            result.push({
              type: "tool",
              content: "User ignored this email draft. Ignore this email and end the workflow.",
              tool_call_id: callId
            });
            processedToolCallIds.add(callId);
            processedOneToolCall = true;
            
            // Go to END
            goto = END;
            
            // Update memory by reflecting on the email tool call
            await updateMemory(
              store,
              ["email_assistant", "triage_preferences"],
              [
                ...state.messages,
                ...result,
                {
                  type: "human",
                  content: `The user ignored the email draft. That means they did not want to respond to the email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: ${MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.`
                }
              ]
            );
          } else if (toolCall.name === "schedule_meeting") {
            // Don't execute the tool, and tell the agent how to proceed
            result.push({
              type: "tool",
              content: "User ignored this calendar meeting draft. Ignore this email and end the workflow.",
              tool_call_id: callId
            });
            processedToolCallIds.add(callId);
            processedOneToolCall = true;
            
            // Go to END
            goto = END;
            
            // Update the memory by reflecting on the full message history including the schedule_meeting tool call
            await updateMemory(
              store,
              ["email_assistant", "triage_preferences"],
              [
                ...state.messages,
                ...result,
                {
                  type: "human",
                  content: `The user ignored the calendar meeting draft. That means they did not want to schedule a meeting for this email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: ${MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.`
                }
              ]
            );
          } else if (toolCall.name === "Question") {
            // Don't execute the tool, and tell the agent how to proceed
            result.push({
              type: "tool",
              content: "User ignored this question. Ignore this email and end the workflow.",
              tool_call_id: callId
            });
            processedToolCallIds.add(callId);
            processedOneToolCall = true;
            
            // Go to END
            goto = END;
            
            // Update the memory by reflecting on the full message history including the Question tool call
            await updateMemory(
              store,
              ["email_assistant", "triage_preferences"],
              [
                ...state.messages,
                ...result,
                {
                  type: "human",
                  content: `The user ignored the Question. That means they did not want to answer the question or deal with this email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: ${MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.`
                }
              ]
            );
          }
        }
      } catch (error: any) {
        console.error(`Error processing tool ${toolCall.name}:`, error);
        result.push({
          type: "tool",
          content: `Error processing tool: ${error.message}`,
          tool_call_id: callId
        });
        processedToolCallIds.add(callId);
        processedOneToolCall = true;
      }
    }
    
    // Return Command with goto and update
    return new Command({
      goto,
      update: {
        messages: result
      }
    });
  } 
};

/**
 * Initialize and export email assistant components
 */
// Set up memory store directly
export const memoryStore = new InMemoryStore();

// Initialize and export the agent graph
export const initializeEmailAssistant = async () => {
  // Setup LLMs and tools
  const { llm, llmWithTools, tools, toolsByName } = await setupLLMAndTools();
  
  // Create the nodes
  const triageRouterNode = createTriageRouterNode(llm, memoryStore);
  const triageInterruptHandlerNode = createTriageInterruptHandlerNode(memoryStore);
  const llmCallNode = createLLMCallNode(llmWithTools, memoryStore);
  const interruptHandlerNode = createInterruptHandlerNode(toolsByName, memoryStore);
  
  // Build agent subgraph
  const agentBuilder = new StateGraph(EmailAgentHITLState)
    .addNode("llm_call", llmCallNode)
    .addNode("interrupt_handler", interruptHandlerNode)
    .addEdge(START, "llm_call")
    .addConditionalEdges(
      "llm_call",
      shouldContinue,
      {
        "interrupt_handler": "interrupt_handler",
        [END]: END
      }
    )
    .addEdge("interrupt_handler", "llm_call");
  
  // Compile the agent with memory store
  const responseAgent = agentBuilder.compile({
    store: memoryStore
  });
  
  // Build overall workflow
  const emailAssistantGraph = new StateGraph(EmailAgentHITLState)
    .addNode("triage_router", triageRouterNode)
    .addNode("triage_interrupt_handler", triageInterruptHandlerNode)
    .addNode("response_agent", responseAgent)
    .addEdge(START, "triage_router")
    // Define conditional edge from triage_router
    .addConditionalEdges(
      "triage_router",
      (state) => {
        const classification = state.classification_decision;
        if (classification === "respond") {
          return "response_agent";
        } else if (classification === "notify") {
          return "triage_interrupt_handler";
        } else {
          return END;
        }
      },
      {
        "response_agent": "response_agent",
        "triage_interrupt_handler": "triage_interrupt_handler",
        [END]: END
      }
    )
    // Define conditional edge from triage_interrupt_handler
    .addConditionalEdges(
      "triage_interrupt_handler",
      (state) => {
        // We use the messages to determine where to go next
        const messages = state.messages;
        if (!messages || messages.length === 0) return END;
        
        // Look for feedback in the messages that indicates responding
        const lastMessage = messages[messages.length - 1];
        if (lastMessage.content && typeof lastMessage.content === 'string' && 
            lastMessage.content.includes("User wants to reply to the email")) {
          return "response_agent";
        }
        return END;
      },
      {
        "response_agent": "response_agent",
        [END]: END
      }
    )
    // Add a simple edge since the response_agent already has the proper conditionals built in
    .addEdge("response_agent", END);
    
  // Compile and return the email assistant
  return emailAssistantGraph.compile({
    store: memoryStore,
    checkpointer: new MemorySaver()
  });
};

// Initialize and export HITL email assistant with memory directly
export const hitlEmailAssistantWithMemory = initializeEmailAssistant();