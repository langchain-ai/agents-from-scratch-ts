/**
 * @fileoverview Email Assistant with Human-in-the-Loop (HITL)
 *
 * This script implements an email assistant with human review capabilities,
 * allowing users to review, edit, or reject proposed actions before execution.
 *
 * @module email_assistant_hitl
 *
 * @structure
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                         HITL Email Assistant                             │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │ COMPONENTS                                                               │
 * │ - LLM                   : GPT-4 model for decision making                │
 * │          │
 * │                                                                          │
 * │ GRAPH NODES                                                              │
 * │ - triage_router         : Classifies emails (ignore/respond/notify)      │
 * │ - triage_interrupt_handler: Gets human input on notification emails      │
 * │ - response_agent        : Subgraph for handling email responses          │
 * │   ├─ llm_call           : Generates responses or tool calls              │
 * │   └─ interrupt_handler  : Handles human review of agent actions          │
 * │                                                                          │
 * │ EDGES                                                                    │
 * │ - START → triage_router                                                  │
 * │ - triage_router → triage_interrupt_handler/response_agent/END            │
 * │ - triage_interrupt_handler → response_agent/END                          │
 * │ - response_agent → END                                                   │
 * │                                                                          │
 * │ KEY FUNCTIONS                                                            │
 * │ - initializeHitlEmailAssistant(): Creates and configures the agent graph │
 * │ - llmCallNode()         : Generates responses using LLM                  │
 * │ - interruptHandlerNode(): Handles human review of tool calls             │
 * │ - triageRouterNode()    : Classifies incoming emails                     │
 * │ - triageInterruptHandlerNode(): Processes notifications with human input │
 * │ - shouldContinue()      : Routes graph based on agent output             │
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
  interrupt,
  Messages,
  LangGraphRunnableConfig,
} from "@langchain/langgraph";
import { ToolCall } from "@langchain/core/messages/tool";

// Zod imports
import "@langchain/langgraph/zod";

// LOCAL IMPORTS
import { getTools, getToolsByName } from "./tools/base.js";
import {
  HITL_TOOLS_PROMPT,
  triageSystemPrompt,
  triageUserPrompt,
  agentSystemPromptHitl,
  defaultBackground,
  defaultResponsePreferences,
  defaultCalPreferences,
  defaultTriageInstructions,
} from "./prompts.js";
import { EmailAgentHITLState, EmailAgentHITLStateType } from "./schemas.js";
import { parseEmail, formatEmailMarkdown, formatForDisplay } from "./utils.js";

// Message Types from LangGraph SDK
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

/**
 * Initialize and export the HITL email assistant
 */
export const initializeHitlEmailAssistant = async () => {
  // Get tools
  const tools = await getTools();
  const toolsByName = await getToolsByName();

  // Initialize the LLM
  const llm = await initChatModel("openai:gpt-4");

  // Initialize the LLM instance for tool use
  const llmWithTools = llm.bindTools(tools, { toolChoice: "required" });

  // Create the LLM call node
  const llmCallNode = async (
    state: EmailAgentHITLStateType,
    config: LangGraphRunnableConfig,
  ) => {
    const { messages } = state;

    // Set up system prompt for the agent
    const systemPrompt = agentSystemPromptHitl
      .replace("{tools_prompt}", HITL_TOOLS_PROMPT)
      .replace("{background}", defaultBackground)
      .replace("{response_preferences}", defaultResponsePreferences)
      .replace("{calendar_preferences}", defaultCalPreferences);

    // Create full message history for the agent
    const allMessages: BaseMessageLike[] = [
      { role: "system", content: systemPrompt },
      ...messages,
    ];

    try {
      // Run the LLM with the messages
      const result = await llmWithTools.invoke(allMessages);

      // Return the AIMessage result - need to cast through unknown since the types don't have proper overlap
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

  // Create the interrupt handler node for human review
  const interruptHandlerNode = async (
    state: EmailAgentHITLStateType,
    config: LangGraphRunnableConfig,
  ): Promise<Command> => {
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
      if (processedOneToolCall) {
        break;
      }
      const callId = toolCall.id ?? `fallback-id-${Date.now()}`;
      const hitlTools = ["write_email", "schedule_meeting", "question"];

      if (!hitlTools.includes(toolCall.name)) {
        const tool = toolsByName[toolCall.name];
        if (!tool) {
          console.error(`Tool ${toolCall.name} not found`);
          result.push({
            role: "tool",
            content: `Error: Tool ${toolCall.name} not found`,
            tool_call_id: callId,
          });
        } else {
          try {
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
          } catch (error: any) {
            console.error(`Error executing tool ${toolCall.name}:`, error);
            result.push({
              role: "tool",
              content: `Error executing tool: ${error.message}`,
              tool_call_id: callId,
            });
          }
        }
        processedToolCallIds.add(callId);
        processedOneToolCall = true;
        continue; // Move to the next tool call or end of loop
      }

      // For HITL tools, proceed to interrupt (NO try...catch around this part)
      const emailInput = state.email_input;
      const parseResult = parseEmail(emailInput);
      if (!parseResult || typeof parseResult !== "object") {
        throw new Error("Invalid email parsing result"); // Or handle more gracefully
      }
      const { author, to, subject, emailThread } = parseResult;
      const originalEmailMarkdown = formatEmailMarkdown(
        subject,
        author,
        to,
        emailThread,
      );
      const toolDisplay = formatForDisplay(toolCall);
      const description = originalEmailMarkdown + toolDisplay;
      const isEditOrAccept =
        toolCall.name === "write_email" || toolCall.name === "schedule_meeting";

      // DO NOT wrap interrupt() in a try...catch block here
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

      // Handle different actions based on the type
      if (reviewAction === "response" && typeof reviewData === "string") {
        if (toolCall.name === "write_email") {
          result.push({
            role: "tool",
            content: `User gave feedback, which can we incorporate into the email. Feedback: ${reviewData}`,
            tool_call_id: callId,
          });
        } else if (toolCall.name === "schedule_meeting") {
          result.push({
            role: "tool",
            content: `User gave feedback, which can we incorporate into the meeting request. Feedback: ${reviewData}`,
            tool_call_id: callId,
          });
        } else if (toolCall.name === "question") {
          result.push({
            role: "tool",
            content: `User answered the question, which can we can use for any follow up actions. Feedback: ${reviewData}`,
            tool_call_id: callId,
          });
        } else {
          // This should not happen if toolCall.name is one of the hitlTools
          console.warn(
            `Unexpected tool ${toolCall.name} in HITL response block.`,
          );
          result.push({
            role: "tool",
            content: `Unexpected tool: ${toolCall.name}`,
            tool_call_id: callId,
          });
        }
        processedToolCallIds.add(callId);
        processedOneToolCall = true; // Mark as processed
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
      } else if (
        reviewAction === "edit" &&
        typeof reviewData === "object" &&
        reviewData !== null
      ) {
        const tool = toolsByName[toolCall.name];
        // Assuming reviewData directly contains the new args for the tool
        const observation = await tool.invoke(reviewData);
        result.push({
          role: "tool",
          content: observation,
          tool_call_id: callId,
        });
        processedToolCallIds.add(callId);
        processedOneToolCall = true;
      } else if (reviewAction === "ignore") {
        // Simplified ignore logic for HITL (no memory updates)
        result.push({
          role: "tool",
          content: `User chose to ignore this ${toolCall.name} action.`,
          tool_call_id: callId,
        });
        processedToolCallIds.add(callId);
        processedOneToolCall = true;
        goto = END; // Or specific handling per tool type if needed
      } else {
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

      // Ensure that after a HITL tool is processed (interrupted and handled), we break the loop.
      if (processedOneToolCall) {
        break;
      }
    }

    // Fallback responses for any remaining tool calls not processed due to the one-HITL-at-a-time logic
    if (lastMessage.tool_calls) {
      for (const tc of lastMessage.tool_calls) {
        const tc_callId = tc.id ?? `fallback-id-${Date.now()}`;
        if (!processedToolCallIds.has(tc_callId)) {
          result.push({
            role: "tool",
            content:
              "Tool execution deferred or pending subsequent review step.",
            tool_call_id: tc_callId,
          });
        }
      }
    }

    return new Command({
      goto,
      update: { messages: result },
    });
  };

  // Conditional routing function
  const shouldContinue = (state: EmailAgentHITLStateType) => {
    const messages = state.messages;
    if (!messages || messages.length === 0) return END;

    const lastMessage = messages[messages.length - 1];

    if (hasToolCalls(lastMessage)) {
      // Check if any tool call is the "Done" tool
      if (lastMessage.tool_calls.some((toolCall) => toolCall.name === "Done")) {
        return END;
      }
      return "interrupt_handler";
    }

    return END;
  };

  // Create the triage router node
  const triageRouterNode = async (
    state: EmailAgentHITLStateType,
    config: LangGraphRunnableConfig,
  ) => {
    try {
      const { email_input } = state;
      const parseResult = parseEmail(email_input);

      // Validate parsing result
      if (!parseResult || typeof parseResult !== "object") {
        throw new Error("Invalid email parsing result");
      }

      const { author, to, subject, emailThread } = parseResult;

      const systemPrompt = triageSystemPrompt
        .replace("{background}", defaultBackground)
        .replace("{triage_instructions}", defaultTriageInstructions);

      const userPrompt = triageUserPrompt
        .replace("{author}", author)
        .replace("{to}", to)
        .replace("{subject}", subject)
        .replace("{email_thread}", emailThread);

      // Create email markdown for Agent Inbox in case of notification
      const emailMarkdown = formatEmailMarkdown(
        subject,
        author,
        to,
        emailThread,
      );

      // Use a simplified approach for classification to avoid structured output issues
      const response = await llm.invoke([
        { role: "system", content: systemPrompt },
        { role: "human", content: userPrompt },
      ]);

      // Extract the classification from the response text
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
            messages: [
              new HumanMessage({
                content: `Respond to the email: ${emailMarkdown}`,
              }),
            ],
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

      // Default to notify if classification is not recognized (shouldn't reach here)
      console.log("❓ Classification: UNKNOWN - Treating as NOTIFY");
      return new Command({
        goto: "triage_interrupt_handler",
        update: {
          classification_decision: "notify",
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

  // Create the triage interrupt handler node
  const triageInterruptHandlerNode = async (
    state: EmailAgentHITLStateType,
    config: LangGraphRunnableConfig,
  ) => {
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
    const initialMessages = [
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

    const returnMessages: Messages = [...initialMessages];
    const reviewAction = humanReview.type;
    const reviewData = humanReview.args;

    if (reviewAction === "response" && reviewData) {
      returnMessages.push({
        role: "human",
        content: `User wants to reply to the email. Use this feedback to respond: ${reviewData}`,
      });
      return new Command({
        goto: "response_agent",
        update: { messages: returnMessages },
      });
    }

    if (reviewAction === "ignore") {
      return new Command({
        goto: END,
        update: { messages: returnMessages },
      });
    }

    // Fallback for unhandled actions
    console.warn(
      `Unknown or unhandled review action in triage: ${reviewAction}`,
    );
    returnMessages.push({
      role: "system",
      content: `Unhandled review action: ${reviewAction}. Ending triage.`,
    });
    return new Command({
      goto: END,
      update: { messages: returnMessages },
    });
  };

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

  // Compile the agent with the checkpointer
  const responseAgent = agentBuilder.compile();

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
    // Add the missing edge from response_agent to END, similar to the memory version
    .addEdge("response_agent", END);

  console.log("Compiling HITL email assistant");

  // Compile and return the email assistant
  return emailAssistantGraph.compile();
};

// Initialize and export HITL email assistant directly with a default checkpointer
export const hitlEmailAssistant = initializeHitlEmailAssistant();
