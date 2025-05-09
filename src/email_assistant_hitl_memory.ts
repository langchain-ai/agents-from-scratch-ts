/**
 * @fileoverview Email Assistant with HITL and Memory
 *
 * This script implements an advanced email assistant that builds on the HITL version
 * by adding persistent memory capabilities to learn from user feedback and preferences.
 *
 * @module email_assistant_hitl_memory
 *
 * @structure
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                       Email Assistant with Memory                        │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │ COMPONENTS                                                               │
 * │ - LLM (with tools)        : GPT-4o model for decision making             │
 * │ - Memory System           : InMemoryStore for maintaining preferences    │
 * │                                                                          │
 * │ GRAPH NODES                                                              │
 * │ - triage_router           : Classifies emails using memory preferences   │
 * │ - triage_interrupt_handler: Gets human feedback on notification emails   │
 * │ - response_agent          : Subgraph for handling email responses        │
 * │   ├─ llm_call             : Generates responses or tool calls            │
 * │   └─ interrupt_handler    : Handles human review of agent actions        │
 * │                                                                          │
 * │ MEMORY NAMESPACES                                                        │
 * │ - ["email_assistant", "triage_preferences"] : Email classification rules │
 * │ - ["email_assistant", "response_preferences"]: Email writing preferences │
 * │ - ["email_assistant", "cal_preferences"]     : Calendar preferences      │
 * │                                                                          │
 * │ KEY FUNCTIONS                                                            │
 * │ - setupLLMAndTools()      : Initializes LLM and tools                    │
 * │ - getMemory()             : Retrieves preferences from memory store      │
 * │ - updateMemory()          : Updates preferences based on user feedback   │
 * │ - triageRouterNode()      : Creates email classification node            │
 * │ - triageInterruptHandlerNode(): Gets human input on notification emails  │
 * │ - interruptHandlerNode()  : Handles human review of agent actions        │
 * │ - llmCallNode()           : Creates node for LLM response generation     │
 * │ - initializeEmailAssistant(): Creates the agent graph with all nodes     │
 * │ - shouldContinue()        : Routes graph based on agent output           │
 * └─────────────────────────────────────────────────────────────────────────┘
 */

// LangChain imports for chat models
import { initChatModel } from "langchain/chat_models/universal";

// LangGraph imports
import {
  StateGraph,
  START,
  END,
  Command,
  MemorySaver,
  InMemoryStore,
  BaseStore,
  LangGraphRunnableConfig,
  interrupt,
  Messages,
  addMessages,
} from "@langchain/langgraph";
import { ToolCall } from "@langchain/core/messages/tool";
import { StructuredTool } from "@langchain/core/tools";

// Zod imports
import "@langchain/langgraph/zod";

// LOCAL IMPORTS
import { getTools, getToolsByName } from "./tools/base.js";
import {
  HITL_MEMORY_TOOLS_PROMPT,
  triageSystemPrompt,
  triageUserPrompt,
  agentSystemPromptHitlMemory,
  defaultBackground,
  defaultResponsePreferences,
  defaultCalPreferences,
  defaultTriageInstructions,
} from "./prompts.js";
import { EmailAgentHITLState, EmailAgentHITLStateType } from "./schemas.js";
import { parseEmail, formatEmailMarkdown, formatForDisplay } from "./utils.js";
import {
  AIMessage,
  BaseMessage,
  BaseMessageLike,
  HumanMessage,
} from "@langchain/core/messages";
import { HumanInterrupt, HumanResponse } from "@langchain/langgraph/prebuilt";

// Helper for type checking
const hasToolCalls = (
  message: BaseMessage,
): message is AIMessage & { tool_calls: ToolCall[] } => {
  return (
    message.getType() === "ai" &&
    "tool_calls" in message &&
    Array.isArray(message.tool_calls) &&
    message.tool_calls.length > 0
  );
};

// Define UserPreferences interface for memory updates
interface UserPreferences {
  preferences: string;
  justification: string;
}

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

// Define the shouldContinue function needed for conditional edges
const shouldContinue = (state: EmailAgentHITLStateType) => {
  const messages = state.messages;
  if (!messages || messages.length === 0) return END;

  const lastMessage = messages[messages.length - 1];

  if (
    hasToolCalls(lastMessage) &&
    lastMessage.tool_calls &&
    lastMessage.tool_calls.length > 0
  ) {
    // Check if any tool call is the "Done" tool
    if (lastMessage.tool_calls.some((toolCall) => toolCall.name === "Done")) {
      return END;
    }
    return "interrupt_handler";
  }

  return END;
};

/**
 *   LLM and tools
 */
export const setupLLMAndTools = async () => {
  // Get tools
  const tools = await getTools();
  const toolsByName = await getToolsByName();

  // Initialize the LLM for use with router / structured output
  const llm = await initChatModel("openai:gpt-4o");

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
  defaultContent: string = "",
): Promise<string> {
  try {
    // Search for existing memory with namespace and key
    const userPreferences = await store.get(namespace, "user_preferences");

    // If memory exists, return its content (the value)
    if (userPreferences) {
      return userPreferences.value.memoryContent;
    }

    // If memory doesn't exist, add it to the store with default content
    await store.put(namespace, "user_preferences", {
      memoryContent: defaultContent,
    });
    return defaultContent;
  } catch (error) {
    console.error("Error getting memory:", error);
    // Return default content if there's an error
    await store.put(namespace, "user_preferences", {
      memoryContent: defaultContent,
    });
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
  store: BaseStore,
  namespace: string[],
  messages: BaseMessage[],
): Promise<void> {
  try {
    // Get the existing memory
    const userPreferences = await store.get(namespace, "user_preferences");
    const currentProfile = userPreferences
      ? userPreferences.value.memoryContent
      : "";

    // Convert Message[] to format expected by the LLM
    const formattedMessages = messages.map((msg) => ({
      role:
        msg.getType() === "human"
          ? "user"
          : msg.getType() === "ai"
            ? "assistant"
            : "system",
      content:
        typeof msg.content === "string"
          ? msg.content
          : JSON.stringify(msg.content),
    }));

    // Initialize chat model
    const llm = await initChatModel("openai:gpt-4o");
    const llmWithStructuredOutput = llm.withStructuredOutput<UserPreferences>({
      schema: {
        role: "object",
        properties: {
          preferences: { role: "string" },
          justification: { role: "string" },
        },
        required: ["preferences"],
      },
    });

    // Create system message with memory update instructions
    const systemMsg = {
      role: "system",
      content: MEMORY_UPDATE_INSTRUCTIONS.replace(
        "{current_profile}",
        currentProfile,
      ).replace("{namespace}", namespace.join("/")),
    };

    // Create user message instructing to update memory
    const userMsg = {
      role: "human",
      content:
        "Think carefully and update the memory profile based upon these user messages:",
    };

    // Convert the formatted messages to Message objects
    const messagesForLLM = formattedMessages.map((msg) => {
      if (msg.role === "user") {
        return { role: "human", content: msg.content };
      } else if (msg.role === "assistant") {
        return { role: "ai", content: msg.content, tool_calls: [] };
      } else {
        return { role: "system", content: msg.content };
      }
    });

    // Invoke the LLM with the messages
    const result = await llmWithStructuredOutput.invoke([
      systemMsg,
      userMsg,
      ...messagesForLLM,
    ]);

    // Save the updated memory to the store
    await store.put(namespace, "user_preferences", {
      memoryContent: result.preferences,
    });
  } catch (error) {
    console.error("Error updating memory:", error);
  }
}

/**
 * Create the triage router node with memory integration
 */
export const triageRouterNode = async (
  state: EmailAgentHITLStateType,
  config: LangGraphRunnableConfig,
): Promise<Command> => {
  const { llm } = await setupLLMAndTools();
  const { store } = config;
  if (!store) {
    throw new Error("Store is required for triage router node");
  }

  try {
    const { email_input } = state;
    const parseResult = parseEmail(email_input);

    // Validate parsing result
    if (!parseResult || typeof parseResult !== "object") {
      throw new Error("Invalid email parsing result");
    }

    const { author, to, subject, emailThread } = parseResult;

    // Get triage preferences from memory
    const triagePreferences = await getMemory(
      store,
      ["email_assistant", "triage_preferences"],
      defaultTriageInstructions,
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
      { role: "system", content: simplifiedSystemPrompt },
      { role: "human", content: userPrompt },
    ]);

    // Extract the classification from the simple response
    let classification: "ignore" | "respond" | "notify";
    const responseText = (response.content || "")
      .toString()
      .toLowerCase()
      .trim();

    if (responseText.includes("respond")) {
      classification = "respond";
    } else if (responseText.includes("notify")) {
      classification = "notify";
    } else if (responseText.includes("ignore")) {
      classification = "ignore";
    } else {
      console.log(
        `Unrecognized classification: "${responseText}". Defaulting to notify.`,
      );
      classification = "notify";
    }

    if (classification === "respond") {
      console.log(
        "📧 Classification: RESPOND - This email requires a response",
      );
      return new Command({
        goto: "response_agent",
        update: {
          classification_decision: classification,
          messages: {
            role: "human",
            content: `Respond to the email: ${emailMarkdown}`,
          },
        },
      });
    } else if (classification === "notify") {
      console.log(
        "🔔 Classification: NOTIFY - This email contains important information",
      );
      return new Command({
        goto: "triage_interrupt_handler",
        update: {
          classification_decision: classification,
        },
      });
    } else if (classification === "ignore") {
      console.log(
        "🚫 Classification: IGNORE - This email can be safely ignored",
      );
      return new Command({
        goto: END,
        update: {
          classification_decision: classification,
        },
      });
    }

    // Default to "ignore" if classification is not recognized
    console.log("❓ Classification: UNKNOWN - Treating as IGNORE");
    return new Command({
      goto: END,
      update: {
        classification_decision: "ignore",
      },
    });
  } catch (error: any) {
    console.error("Error in triage router:", error);
    return new Command({
      goto: END,
      update: {
        classification_decision: "error",
        messages: [
          {
            role: "system",
            content: `Error in triage router: ${error.message}`,
          },
        ],
      },
    });
  }
};

/**
 * Create the triage interrupt handler node with memory integration
 */
export const triageInterruptHandlerNode = async (
  state: EmailAgentHITLStateType,
  config: LangGraphRunnableConfig,
): Promise<Command> => {
  const { store } = config;
  if (!store) {
    throw new Error("Store is required for triage interrupt handler node");
  }

  // Parse the email input
  const parseResult = parseEmail(state.email_input);

  // Validate parsing result
  if (!parseResult || typeof parseResult !== "object") {
    throw new Error("Invalid email parsing result");
  }

  const { author, to, subject, emailThread } = parseResult;

  // Create email markdown for Agent Inbox in case of notification
  const emailMarkdown = formatEmailMarkdown(subject, author, to, emailThread);

  // Create messages
  const messages = [
    {
      role: "human",
      content: `Email to notify user about: ${emailMarkdown}`,
    },
  ];

  // DO NOT wrap interrupt() in a try...catch block here
  const humanReview = interrupt<HumanInterrupt, HumanResponse[]>({
    action_request: {
      action: `Email requires attention: ${state.classification_decision || "notify"}`,
      args: {},
    },
    description: emailMarkdown,
    config: {
      allow_ignore: true,
      allow_respond: true,
      allow_edit: false,
      allow_accept: false,
    },
  })[0];

  // Initialize return values
  const returnMessages: Messages = [...messages];

  // Handle different response types
  const reviewAction = humanReview.type;
  const reviewData = humanReview.args;

  if (reviewAction === "response" && reviewData) {
    returnMessages.push({
      role: "human",
      content: `User wants to reply to the email. Use this feedback to respond: ${reviewData}`,
    });
    // Update memory with feedback
    const memoryUpdateMessages = [
      {
        role: "human",
        content:
          "The user decided to respond to the email, so update the triage preferences to capture this.",
      },
      ...returnMessages,
    ];

    await updateMemory(
      store,
      ["email_assistant", "triage_preferences"],
      addMessages([], memoryUpdateMessages),
    );

    return new Command({
      goto: "response_agent",
      update: {
        messages: returnMessages,
      },
    });
  }

  if (reviewAction === "ignore") {
    // User ignored the email or other action
    const memoryUpdateMessages = [
      {
        role: "human",
        content:
          "The user decided to ignore the email even though it was classified as notify. Update triage preferences to capture this.",
      },
      ...returnMessages,
    ];

    // Update memory with feedback
    await updateMemory(
      store,
      ["email_assistant", "triage_preferences"],
      addMessages([], memoryUpdateMessages),
    );

    return new Command({
      goto: END,
      update: {
        messages: returnMessages,
      },
    });
  }

  // If we reach here, it means reviewAction was not 'response' or 'ignore'
  // This could be an unhandled action type from the interrupt or an issue.
  // For robustness, we can log this and default to ending the process.
  console.warn(`Unknown or unhandled review action in triage: ${reviewAction}`);
  return new Command({
    goto: END,
    update: {
      messages: [
        ...returnMessages,
        {
          role: "system",
          content: `Unhandled review action: ${reviewAction}. Ending triage.`,
        },
      ],
    },
  });
};

/**
 * Create the LLM call node with memory integration
 */
export const llmCallNode = async (
  state: EmailAgentHITLStateType,
  config: LangGraphRunnableConfig,
) => {
  const { llmWithTools } = await setupLLMAndTools();

  const { store } = config;
  if (!store) {
    throw new Error("Store is required for LLM call node");
  }

  try {
    // Get memory preferences
    const calPreferences = await getMemory(
      store,
      ["email_assistant", "cal_preferences"],
      defaultCalPreferences,
    );

    const responsePreferences = await getMemory(
      store,
      ["email_assistant", "response_preferences"],
      defaultResponsePreferences,
    );

    // Set up system prompt for the agent
    const systemPrompt = agentSystemPromptHitlMemory
      .replace("{tools_prompt}", HITL_MEMORY_TOOLS_PROMPT)
      .replace("{background}", defaultBackground)
      .replace("{response_preferences}", responsePreferences)
      .replace("{cal_preferences}", calPreferences);

    // Create full message history for the agent
    const allMessages = [
      { role: "system", content: systemPrompt },
      ...state.messages,
    ];

    // Run the LLM with the messages
    const result = await llmWithTools.invoke(allMessages);

    // Return the AIMessage result
    return {
      messages: result,
    };
  } catch (error: any) {
    console.error("Error in LLM call:", error);
    return {
      messages: [
        {
          role: "ai",
          content: `Error calling model: ${error.message}`,
        },
      ],
    };
  }
};

/**
 * Create the interrupt handler node with memory updates
 */
export const interruptHandlerNode = async (
  state: EmailAgentHITLStateType,
  config: LangGraphRunnableConfig,
): Promise<Command> => {
  const { toolsByName } = await setupLLMAndTools();
  const { store } = config;
  if (!store) {
    throw new Error("Store is required for interrupt handler node");
  }

  // Store messages to be returned
  const result: BaseMessageLike[] = [];

  // Default goto is llm_call
  let goto: typeof END | "llm_call" = "llm_call";

  // Get the last message
  const lastMessage = state.messages[state.messages.length - 1];

  // Exit early if there are no tool calls
  if (
    !hasToolCalls(lastMessage) ||
    !lastMessage.tool_calls ||
    lastMessage.tool_calls.length === 0
  ) {
    return new Command({
      goto,
      update: { messages: [] },
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
      break;
    }

    // Get or create a valid tool call ID
    const callId = toolCall.id ?? `fallback-id-${Date.now()}`;

    // Allowed tools for HITL
    const hitlTools = ["write_email", "schedule_meeting", "question"];

    // If tool is not in our HITL list, execute it directly without interruption
    if (!hitlTools.includes(toolCall.name)) {
      const tool = toolsByName[toolCall.name];
      if (!tool) {
        console.error(`Tool ${toolCall.name} not found`);
        result.push({
          role: "tool",
          content: `Error: Tool ${toolCall.name} not found`,
          tool_call_id: callId,
        });
        processedToolCallIds.add(callId);
        processedOneToolCall = true;
        continue; // Important: continue to next tool_call or to the end of loop
      }

      try {
        // Parse the args properly - if it's a string, parse it as JSON
        const parsedArgs =
          typeof toolCall.args === "string"
            ? JSON.parse(toolCall.args)
            : toolCall.args;

        // Invoke the tool with properly formatted arguments
        const observation = await tool.invoke(parsedArgs);
        result.push({
          role: "tool",
          content: observation,
          tool_call_id: callId,
        });
        processedToolCallIds.add(callId);
        processedOneToolCall = true;
        continue; // Important: continue to next tool_call or to the end of loop
      } catch (error: any) {
        console.error(`Error executing tool ${toolCall.name}:`, error);
        result.push({
          role: "tool",
          content: `Error executing tool: ${error.message}`,
          tool_call_id: callId,
        });
        processedToolCallIds.add(callId);
        processedOneToolCall = true;
        continue; // Important: continue to next tool_call or to the end of loop
      }
    }

    // Get original email from email_input in state
    const emailInput = state.email_input;
    const parseResult = parseEmail(emailInput);
    // Validate parsing result
    if (!parseResult || typeof parseResult !== "object") {
      throw new Error("Invalid email parsing result");
    }
    const { author, to, subject, emailThread } = parseResult;
    const originalEmailMarkdown = formatEmailMarkdown(
      subject,
      author,
      to,
      emailThread,
    );

    // Format tool call for display and prepend the original email
    const toolDisplay = formatForDisplay(toolCall);
    const description = originalEmailMarkdown + toolDisplay;

    // DO NOT wrap interrupt() in a try...catch block here
    const isEditOrAccept =
      toolCall.name === "write_email" || toolCall.name === "schedule_meeting";
    const humanReview = interrupt<HumanInterrupt, HumanResponse[]>({
      action_request: {
        action: `Review this ${toolCall.name} action:`,
        args: toolCall.args,
      },
      description,
      config: {
        allow_ignore: true,
        allow_respond: true,
        allow_edit: isEditOrAccept,
        allow_accept: isEditOrAccept,
      },
    })[0];

    const reviewAction = humanReview.type;
    const reviewData = humanReview.args;

    if (reviewAction === "response" && typeof reviewData === "string") {
      if (toolCall.name === "write_email") {
        // Don't execute the tool, and add a message with the user feedback to incorporate into the email
        result.push({
          role: "tool",
          content: `User gave feedback, which can we incorporate into the email. Feedback: ${reviewData}`,
          tool_call_id: toolCall.id,
        });
        await updateMemory(
          store,
          ["email_assistant", "response_preferences"],
          addMessages(
            [],
            [
              ...state.messages,
              ...result,
              {
                role: "human",
                content: `User gave feedback, which we can use to update the calendar preferences. Follow all instructions above, and remember: ${MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.`,
              },
            ],
          ),
        );
      } else if (toolCall.name === "schedule_meeting") {
        // Don't execute the tool, and add a message with the user feedback to incorporate into the email
        result.push({
          role: "tool",
          content: `User gave feedback, which can we incorporate into the meeting request. Feedback: ${reviewData}`,
          tool_call_id: toolCall.id,
        });
        await updateMemory(
          store,
          ["email_assistant", "response_preferences"],
          addMessages(
            [],
            [
              ...state.messages,
              ...result,
              {
                role: "human",
                content: `User gave feedback, which we can use to update the calendar preferences. Follow all instructions above, and remember: ${MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.`,
              },
            ],
          ),
        );
      } else if (toolCall.name === "question") {
        // Don't execute the tool, and add a message with the user feedback to incorporate into the email
        result.push({
          role: "tool",
          content: `User answered the question, which can we can use for any follow up actions. Feedback: ${reviewData}`,
          tool_call_id: toolCall.id,
        });
      } else {
        throw new Error(`Invalid tool call: ${toolCall.name}`);
      }
      processedToolCallIds.add(callId);
      processedOneToolCall = true;
    } else if (reviewAction === "accept") {
      const tool = toolsByName[toolCall.name];
      const parsedArgs =
        typeof toolCall.args === "string"
          ? JSON.parse(toolCall.args)
          : toolCall.args;
      const observation = await tool.invoke(parsedArgs);
      result.push({
        role: "tool",
        content: observation,
        tool_call_id: callId,
      });
      processedToolCallIds.add(callId);
      processedOneToolCall = true;
      // If a write_email action is accepted, the response agent's job is done.
      if (toolCall.name === "write_email") {
        goto = END;
      }
    } else if (
      reviewAction === "edit" &&
      typeof reviewData === "object" &&
      reviewData !== null
    ) {
      // Tool selection
      const tool = toolsByName[toolCall.name];

      // Save the initial tool call for memory updates
      const initialToolCall = `${toolCall.name}: ${JSON.stringify(toolCall.args, null, 2)}`;

      // Update the AI message's tool call with edited content (reference to the message in the state)
      if (hasToolCalls(lastMessage)) {
        // Replace the original tool call with the edited one
        lastMessage.tool_calls = lastMessage.tool_calls?.map((tc) => {
          if (tc.id === callId) {
            return {
              ...tc,
              args: reviewData.args,
            };
          }
          return tc;
        });
      }

      // Execute the tool with edited args
      const observation = await tool.invoke(reviewData.args);

      // Add the tool response
      result.push({
        role: "tool",
        content: observation,
        tool_call_id: callId,
      });
      processedToolCallIds.add(callId);
      processedOneToolCall = true;

      // Update memory with user edits
      if (toolCall.name === "write_email") {
        await updateMemory(
          store,
          ["email_assistant", "response_preferences"],
          [
            new HumanMessage(
              `User edited the email response. Here is the initial email generated by the assistant: ${initialToolCall}. Here is the edited email: ${JSON.stringify(reviewData.args)}. Follow all instructions above, and remember: ${MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.`,
            ),
          ],
        );
      } else if (toolCall.name === "schedule_meeting") {
        await updateMemory(
          store,
          ["email_assistant", "cal_preferences"],
          [
            new HumanMessage(
              `User edited the calendar invitation. Here is the initial calendar invitation generated by the assistant: ${initialToolCall}. Here is the edited calendar invitation: ${JSON.stringify(reviewData.args)}. Follow all instructions above, and remember: ${MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.`,
            ),
          ],
        );
      }
    } else if (reviewAction === "ignore") {
      if (toolCall.name === "write_email") {
        // Don't execute the tool, and tell the agent how to proceed
        result.push({
          role: "tool",
          content:
            "User ignored this email draft. Ignore this email and end the workflow.",
          tool_call_id: callId,
        });
        processedToolCallIds.add(callId);
        processedOneToolCall = true;

        // Update memory by reflecting on the email tool call
        await updateMemory(
          store,
          ["email_assistant", "triage_preferences"],
          addMessages(
            [],
            [
              ...state.messages,
              ...result,
              {
                role: "human",
                content: `The user ignored the email draft. That means they did not want to respond to the email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: ${MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.`,
              },
            ],
          ),
        );

        goto = END;
      } else if (toolCall.name === "schedule_meeting") {
        // Don't execute the tool, and tell the agent how to proceed
        result.push({
          role: "tool",
          content:
            "User ignored this calendar meeting draft. Ignore this email and end the workflow.",
          tool_call_id: callId,
        });
        processedToolCallIds.add(callId);
        processedOneToolCall = true;

        // Update the memory by reflecting on the full message history including the schedule_meeting tool call
        await updateMemory(
          store,
          ["email_assistant", "triage_preferences"],
          addMessages(
            [],
            [
              ...state.messages,
              ...result,
              {
                role: "human",
                content: `The user ignored the calendar meeting draft. That means they did not want to schedule a meeting for this email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: ${MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.`,
              },
            ],
          ),
        );

        goto = END;
      } else if (toolCall.name === "question") {
        // Don't execute the tool, and tell the agent how to proceed
        result.push({
          role: "tool",
          content:
            "User ignored this question. Ignore this email and end the workflow.",
          tool_call_id: callId,
        });
        processedToolCallIds.add(callId);
        processedOneToolCall = true;

        // Update the memory by reflecting on the full message history including the Question tool call
        await updateMemory(
          store,
          ["email_assistant", "triage_preferences"],
          addMessages(
            [],
            [
              ...state.messages,
              ...result,
              {
                role: "human",
                content: `The user ignored the Question. That means they did not want to answer the question or deal with this email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: ${MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.`,
              },
            ],
          ),
        );

        goto = END;
      }
      processedToolCallIds.add(callId);
      processedOneToolCall = true;
    } else {
      // This case should ideally not be reached if interrupt config is exhaustive
      console.warn(
        `Unhandled review action for tool ${toolCall.name}: ${reviewAction}`,
      );
      result.push({
        role: "tool",
        content: `Unhandled review action: ${reviewAction}`,
        tool_call_id: callId,
      });
      processedToolCallIds.add(callId);
      processedOneToolCall = true; // Mark as processed to move to the next step
    }
    // If a direct return wasn't made (e.g. in ignore case leading to END)
    // ensure we break if one tool call was processed by HITL interrupt.
    if (processedOneToolCall) {
      break; // We break here because we only want to process one HITL tool interrupt at a time
    }
  }
  // If we iterate through all tool_calls and none were HITL / processedOneToolCall is still false,
  // but results were populated by non-HITL tools, this is fine.
  // If results is empty and processedOneToolCall is false, it implies an empty tool_calls array or non-matching tools.

  // If we processed one HITL tool and broke, or if we processed only non-HITL tools,
  // we still need to provide placeholder responses for any remaining tool calls
  // that were not processed due to the one-HITL-at-a-time logic or if they occurred after the break.
  if (lastMessage.tool_calls) {
    for (const tc of lastMessage.tool_calls) {
      const tc_callId = tc.id ?? `fallback-id-${Date.now()}`;
      if (!processedToolCallIds.has(tc_callId)) {
        result.push({
          role: "tool",
          content: "Tool execution deferred or pending subsequent review step.",
          tool_call_id: tc_callId,
        });
      }
    }
  }

  // Return Command with goto and update
  return new Command({
    goto,
    update: { messages: result },
  });
};

/**
 * Initialize and export email assistant components
 */

// Initialize and export the agent graph
export const initializeEmailAssistant = async () => {
  const memoryStore = new InMemoryStore();
  const checkpointer = new MemorySaver();

  // Build agent subgraph
  const agentBuilder = new StateGraph(EmailAgentHITLState)
    .addNode("llm_call", llmCallNode)
    .addNode("interrupt_handler", interruptHandlerNode)
    .addEdge(START, "llm_call")
    .addConditionalEdges("llm_call", shouldContinue, {
      interrupt_handler: "interrupt_handler",
      [END]: END,
    })
    .addEdge("interrupt_handler", "llm_call");

  // Compile the agent with memory store
  const responseAgent = agentBuilder.compile({
    store: memoryStore,
    checkpointer,
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
        response_agent: "response_agent",
        triage_interrupt_handler: "triage_interrupt_handler",
        [END]: END,
      },
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
        if (
          lastMessage.content &&
          typeof lastMessage.content === "string" &&
          lastMessage.content.includes("User wants to reply to the email")
        ) {
          return "response_agent";
        }
        return END;
      },
      {
        response_agent: "response_agent",
        [END]: END,
      },
    )
    // Add a simple edge since the response_agent already has the proper conditionals built in
    .addEdge("response_agent", END);

  // Compile and return the email assistant
  return emailAssistantGraph.compile({
    store: memoryStore,
    checkpointer,
  });
};

// Initialize and export HITL email assistant with memory directly
export const hitlEmailAssistantWithMemory = initializeEmailAssistant();
