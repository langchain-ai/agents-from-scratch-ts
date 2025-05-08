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
 * │ - Checkpointer          : MemorySaver for saving workflow state          │
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
  MemorySaver,
  interrupt,
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
import { z } from "zod";

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
  const llmCallNode = async (state: EmailAgentHITLStateType) => {
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

    // Run the LLM with the messages
    const result = await llmWithTools.invoke(allMessages);

    // Return the AIMessage result - need to cast through unknown since the types don't have proper overlap
    return {
      messages: result,
    };
  };

  // Create the interrupt handler node for human review
  const interruptHandlerNode = async (
    state: EmailAgentHITLStateType,
  ): Promise<Command> => {
    // Store messages to be returned
    const result: BaseMessageLike[] = [];

    // Default goto is llm_call
    let goto: typeof END | "llm_call" = "llm_call";

    // Get the last message
    const lastMessage = state.messages[state.messages.length - 1];

    // Exit early if there are no tool calls
    if (!hasToolCalls(lastMessage)) {
      return new Command({
        goto,
        update: { messages: [] },
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
        } catch (error: any) {
          console.error(`Error executing tool ${toolCall.name}:`, error);
          result.push({
            role: "tool",
            content: `Error executing tool: ${error.message}`,
            tool_call_id: callId,
          });
          processedToolCallIds.add(callId);
          processedOneToolCall = true;
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

      // Format tool call for display
      const toolDisplay = formatForDisplay(toolCall);
      const description = originalEmailMarkdown + toolDisplay;

      // IMPORTANT: We're directly passing the interrupt call result without modifying it
      const humanReview = interrupt<HumanInterrupt, HumanResponse[]>({
        action_request: {
          action: "Review this tool call before execution:",
          args: toolCall.args,
        },
        description,
        config: {
          allow_ignore: true,
          allow_respond: true,
          allow_edit: true,
          allow_accept: true,
        },
      })[0];

      const reviewAction = humanReview.type;
      const reviewData = humanReview.args;

      if (reviewAction === "accept") {
        // Execute the tool with original args
        const tool = toolsByName[toolCall.name];
        // Parse the args properly
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
        reviewData
      ) {
        // Execute with edited args
        const tool = toolsByName[toolCall.name];

        const observation = await tool.invoke(reviewData);
        result.push({
          role: "tool",
          content: observation,
          tool_call_id: callId,
        });
        processedToolCallIds.add(callId);
        processedOneToolCall = true;
      } else if (
        reviewAction === "response" &&
        typeof reviewData === "string"
      ) {
        // Add feedback as a tool message
        result.push({
          role: "tool",
          content: reviewData,
          tool_call_id: callId,
        });
        processedToolCallIds.add(callId);
        processedOneToolCall = true;
        goto = "llm_call";
      } else if (reviewAction === "ignore") {
        // Even when stopping, we still need to respond to the tool call
        result.push({
          role: "tool",
          content: "User chose to ignore this action.",
          tool_call_id: callId,
        });
        processedToolCallIds.add(callId);
        processedOneToolCall = true;
        goto = END;
      } else {
        throw new Error(`Unknown action: ${reviewAction}`);
      }
    }

    //  ----------------------------------------
    // TODO: `processedOneToolCall` does not exist in PY. Remove & update any logic which relies on it
    //  ----------------------------------------
    if (processedOneToolCall) {
      return new Command({
        goto,
        update: { messages: result },
      });
    }

    // If we reach here and haven't processed any tool calls,
    // we need to return appropriate responses for any remaining ones
    lastMessage.tool_calls?.forEach((toolCall) => {
      const callId = toolCall.id ?? `fallback-id-${Date.now()}`;
      if (!processedToolCallIds.has(callId)) {
        // We've skipped this tool call, but we still need to respond to it
        // This is important for OpenAI's API requirement that every tool call has a response
        result.push({
          role: "tool",
          content: "Tool execution pending human review.",
          tool_call_id: callId,
        });
      }
    });

    // Return the Command with goto and update
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
  const triageRouterNode = async (state: EmailAgentHITLStateType) => {
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

      const jsonSystemPrompt = `${systemPrompt}\n\nProvide your response in the following JSON format:
{
  "reasoning": "your step-by-step reasoning",
  "classification": "ignore" | "respond" | "notify"
}`;

      const classificationSchema = z.object({
        reasoning: z.string().describe("your step-by-step reasoning"),
        classification: z
          .enum(["ignore", "respond", "notify"])
          .describe("The classification of the email"),
      });

      const llmWithClassification = llm.withStructuredOutput(
        classificationSchema,
        {
          name: "classification",
        },
      );

      // Use the regular LLM instead of withStructuredOutput
      const response = await llmWithClassification.invoke([
        { role: "system", content: jsonSystemPrompt },
        { role: "human", content: userPrompt },
      ]);

      let goto: "triage_interrupt_handler" | "response_agent" | typeof END =
        END;
      let update: Partial<EmailAgentHITLStateType> = {
        classification_decision: response.classification,
      };

      // Create message
      update.messages = [
        new HumanMessage({ content: `Email to review: ${emailMarkdown}` }),
      ];

      if (response.classification === "respond") {
        console.log(
          "📧 Classification: RESPOND - This email requires a response",
        );
        goto = "response_agent";
      } else if (response.classification === "notify") {
        console.log(
          "🔔 Classification: NOTIFY - This email contains important information",
        );
        goto = "triage_interrupt_handler";
      } else if (response.classification === "ignore") {
        console.log(
          "🚫 Classification: IGNORE - This email can be safely ignored",
        );
        goto = END;
      } else {
        // Default to notify if classification is not recognized
        goto = "triage_interrupt_handler";
        update.classification_decision = "notify";
      }

      return new Command({
        goto,
        update,
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
  const triageInterruptHandlerNode = async (state: EmailAgentHITLStateType) => {
    // Parse the email input
    const parseResult = parseEmail(state.email_input);

    // Validate parsing result
    if (!parseResult || typeof parseResult !== "object") {
      throw new Error("Invalid email parsing result");
    }

    const { author, to, subject, emailThread } = parseResult;

    // Create email markdown for Agent Inbox in case of notification
    const emailMarkdown = formatEmailMarkdown(subject, author, to, emailThread);

    try {
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
          allow_accept: true,
        },
      })[0];

      let goto: "response_agent" | typeof END = END;
      const messages = [
        { role: "human", content: `Email to review: ${emailMarkdown}` },
      ];

      // Handle different response types
      const reviewAction = humanReview.type;
      const reviewData = humanReview.args;

      if (reviewAction === "accept") {
        // Human wants to handle this email - proceed to response agent
        goto = "response_agent";
      } else if (
        reviewAction === "response" &&
        typeof reviewData === "string"
      ) {
        // Human provided feedback or instructions on how to handle
        messages.push({ role: "human", content: reviewData });
        goto = "response_agent";
      } else {
        // Default to END for other actions
        goto = END;
      }

      // Return Command with goto and update
      return new Command({
        goto,
        update: {
          messages,
        },
      });
    } catch (error) {
      console.error("Error with triage interrupt handler:", error);
      return new Command({
        goto: END,
        update: {
          messages: [
            { role: "system", content: `Error in triage interrupt: ${error}` },
          ],
        },
      });
    }
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

  // Compile the agent
  const responseAgent = agentBuilder.compile();

  // Build overall workflow
  const emailAssistantGraph = new StateGraph(EmailAgentHITLState)
    .addNode("triage_router", triageRouterNode, {
      ends: ["triage_interrupt_handler", "response_agent", END],
    })
    .addNode("triage_interrupt_handler", triageInterruptHandlerNode, {
      ends: ["response_agent", END],
    })
    .addNode("response_agent", responseAgent)
    .addEdge(START, "triage_router")
    .addEdge("response_agent", END);

  // Use provided checkpointer or create a new one
  const actualCheckpointer = new MemorySaver();

  console.log(
    "Compiling HITL email assistant with checkpointer:",
    actualCheckpointer ? "provided" : "default",
  );

  // Compile and return the email assistant with the checkpointer
  return emailAssistantGraph.compile({
    checkpointer: actualCheckpointer,
  });
};

// Initialize and export HITL email assistant directly with a default checkpointer
export const hitlEmailAssistant = initializeHitlEmailAssistant();
