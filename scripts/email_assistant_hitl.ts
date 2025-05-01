// LangChain imports for chat models
import { initChatModel } from "langchain/chat_models/universal";

// LangGraph imports
import { 
  StateGraph, 
  START, 
  END,
  Command,
  Annotation,
  interrupt,
  messagesStateReducer
} from "@langchain/langgraph";
import { 
  BaseMessage, 
  HumanMessage, 
  AIMessage, 
  SystemMessage,
  isAIMessage
} from "@langchain/core/messages";
import { ToolCall, ToolMessage } from "@langchain/core/messages/tool";
import { isToolMessage } from "@langchain/core/messages/tool";
import { ToolNode } from "@langchain/langgraph/prebuilt";

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
} from "../lib/schemas.js";
import {
  parseEmail,
  formatEmailMarkdown,
  formatForDisplay
} from "../lib/utils.js";

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
  
  // Define the agent state
  const AgentState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
      reducer: messagesStateReducer,
      default: () => [],
    }),
    email_input: Annotation<EmailData>(),
    classification_decision: Annotation<"ignore" | "respond" | "notify" | "error" | null>(),
  });

  // Define proper TypeScript types for our state
  type AgentStateType = typeof AgentState.State;
  
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
      new SystemMessage({ content: systemPrompt }),
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
    const result: BaseMessage[] = [];
    
    // Default goto is llm_call
    let goto: typeof END | "llm_call" = "llm_call";
    
    // Get the last message
    const lastMessage = state.messages[state.messages.length - 1];
    
    // Exit early if there are no tool calls
    if (!isAIMessage(lastMessage) || !lastMessage.tool_calls || lastMessage.tool_calls.length === 0) {
      return new Command({
        goto,
        update: { messages: result }
      });
    }
    
    // Iterate over the tool calls in the last message
    for (const toolCall of lastMessage.tool_calls) {
      // Allowed tools for HITL
      const hitlTools = ["write_email", "schedule_meeting", "Question"];
      
      // If tool is not in our HITL list, execute it directly without interruption
      if (!hitlTools.includes(toolCall.name)) {
        const tool = toolsByName[toolCall.name];
        // Explicitly convert args to string if needed to satisfy type constraints
        const args = typeof toolCall.args === 'string' ? toolCall.args : JSON.stringify(toolCall.args);
        const observation = await tool.invoke(args);
        const callId = toolCall.id ?? `fallback-id-${Date.now()}`;
        result.push(new ToolMessage({
          content: observation,
          tool_call_id: callId
        }));
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
          const args = typeof toolCall.args === 'string' ? toolCall.args : JSON.stringify(toolCall.args);
          const observation = await tool.invoke(args);
          const callId = toolCall.id ?? `fallback-id-${Date.now()}`;
          result.push(new ToolMessage({
            content: observation,
            tool_call_id: callId
          }));
        } 
        else if (reviewAction === "update") {
          // Execute with edited args
          const tool = toolsByName[toolCall.name];
          const updatedArgs = reviewData;
          const args = typeof updatedArgs === 'string' ? updatedArgs : JSON.stringify(updatedArgs);
          const observation = await tool.invoke(args);
          const callId = toolCall.id ?? `fallback-id-${Date.now()}`;
          result.push(new ToolMessage({
            content: observation,
            tool_call_id: callId
          }));
        }
        else if (reviewAction === "feedback") {
          // Add feedback as a tool message
          const callId = toolCall.id ?? `fallback-id-${Date.now()}`;
          result.push(new ToolMessage({
            content: reviewData,
            tool_call_id: callId
          }));
          goto = "llm_call";
        }
        else {
          goto = END;
        }
      } catch (error) {
        console.error("Error with interrupt handler:", error);
        throw error;
      }
    }
    
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
        new SystemMessage({ content: jsonSystemPrompt }),
        new HumanMessage({ content: userPrompt })
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
        new HumanMessage({
          content: `Email to review: ${emailMarkdown}`
        })
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
            new SystemMessage({ 
              content: `Error in triage router: ${error.message}`
            })
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
      const humanReview = await interrupt({
        question: `Email requires attention: ${state.classification_decision || "notify"}`,
        email: emailMarkdown
      });
      
      let goto: "response_agent" | typeof END = END;
      const messages = [
        new HumanMessage({
          content: `Email to review: ${emailMarkdown}`
        })
      ];
      
      // Handle different response types
      const reviewAction = humanReview.action;
      const reviewData = humanReview.data;
      
      if (reviewAction === "continue") {
        // Human wants to handle this email - proceed to response agent
        goto = "response_agent";
      } else if (reviewAction === "feedback") {
        // Human provided feedback or instructions on how to handle
        messages.push(new HumanMessage({
          content: reviewData
        }));
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
            new SystemMessage({ 
              content: `Error in triage interrupt: ${error}`
            })
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
    
    if (isAIMessage(lastMessage) && lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
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

  // Build agent workflow
  const agentBuilder = new StateGraph<typeof AgentState.spec, 
    AgentStateType, 
    Partial<AgentStateType>,
    AgentNodes>(AgentState);
  
  // Add nodes
  agentBuilder.addNode("llm_call", llmCall);
  agentBuilder.addNode("interrupt_handler", interruptHandler);
  
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
  
  // Build overall workflow
  const overallWorkflow = new StateGraph<typeof AgentState.spec, 
    AgentStateType, 
    Partial<AgentStateType>,
    AgentNodes>(AgentState);
  
  overallWorkflow.addNode("triage_router", triageRouter, {
    ends: ["triage_interrupt_handler", "response_agent", END]
  });
  overallWorkflow.addNode("triage_interrupt_handler", triageInterruptHandler, {
    ends: ["response_agent", END]
  });
  overallWorkflow.addNode("response_agent", responseAgent);
  overallWorkflow.addEdge(START, "triage_router");
  
  // Compile the email assistant
  const hitlEmailAssistant = overallWorkflow.compile();
  
  return hitlEmailAssistant;
};

// For server-side usage
export async function getHitlEmailAssistant() {
  return await createHitlEmailAssistant();
} 