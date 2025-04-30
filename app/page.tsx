import Image from "next/image";
// LangChain imports for chat models
import { initChatModel } from "langchain/chat_models/universal";

// LangGraph imports
import { StructuredTool } from "@langchain/core/tools";
import { 
  StateGraph, 
  START, 
  END,
  Command,
  Annotation,
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
      
      <div className="flex flex-col w-full max-w-3xl gap-8">
        <h2 className="text-2xl font-bold">Email Assistant</h2>
        <p>This page contains the TypeScript implementation of the email assistant workflow.</p>
        
        <div className="mt-4">
          <a 
            href="/test-email-assistant" 
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Test Email Assistant
          </a>
        </div>
      </div>
    </div>
  );
}

// Create the email assistant workflow
export const createEmailAssistant = async () => {
  // Get tools
  const tools = getTools();
  const toolsByName = getToolsByName(tools);
  
  // Initialize the LLM for use with router / structured output
  const llm = await initChatModel("openai:gpt-4", { temperature: 0.0 });
  const llmRouter = llm.withStructuredOutput(RouterSchema);
  
  // Initialize the LLM, enforcing tool use (of any available tools) for agent
  const llmWithTools = llm.bindTools(tools, { toolChoice: "required" });
  
  // Define the agent state
  const AgentState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
      reducer: messagesStateReducer,
      default: () => [],
    }),
    email_input: Annotation<EmailData>(),
    classification_decision: Annotation<"ignore" | "respond" | "notify" | undefined>(),
  });

  // Define proper TypeScript types for our state
  type AgentStateType = typeof AgentState.State;
  
  // Nodes
  const llmCall = async (state: AgentStateType) => {
    /**
     * LLM decides whether to call a tool or not
     * This is the main decision-making node that generates responses or tool calls
     */
    const messages = [...state.messages];
    const systemPromptContent = agentSystemPrompt
      .replace("{tools_prompt}", "AGENT_TOOLS_PROMPT")
      .replace("{background}", defaultBackground)
      .replace("{response_preferences}", defaultResponsePreferences)
      .replace("{cal_preferences}", defaultCalPreferences);
    
    const response = await llmWithTools.invoke([
      new SystemMessage({ content: systemPromptContent }),
      ...messages
    ]);
    
    return {
      messages: [response]
    };
  };
  
  // Create the tool node using LangGraph's ToolNode
  const toolNode = new ToolNode(tools);
  
  // Conditional edge function
  const shouldContinue = (state: AgentStateType) => {
    /**
     * Route to environment for tool execution, or end if Done tool called
     * Similar to the Python version's should_continue function
     */
    const messages = state.messages;
    if (!messages || messages.length === 0) return END;
    
    const lastMessage = messages[messages.length - 1];
    
    if (isAIMessage(lastMessage) && lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
      // Check if any tool call is the "Done" tool
      if (lastMessage.tool_calls.some(toolCall => toolCall.name === "Done")) {
        return END;
      }
      return "environment";
    }
    
    return END;
  };
  
  // Define node names as a union type for better type safety
  type AgentNodes = typeof START | typeof END | "llm_call" | "environment" | "triage_router" | "response_agent";

  // Use the type parameter to specify allowed node names
  const agentBuilder = new StateGraph<typeof AgentState.spec, 
    AgentStateType, 
    Partial<AgentStateType>,
    AgentNodes>(AgentState);
  
  // Add nodes
  agentBuilder.addNode("llm_call", llmCall);
  agentBuilder.addNode("environment", toolNode);
  
  // Add edges to connect nodes
  agentBuilder.addEdge(START, "llm_call");
  agentBuilder.addConditionalEdges(
    "llm_call",
    shouldContinue,
    {
      "environment": "environment",
      [END]: END
    }
  );
  agentBuilder.addEdge("environment", "llm_call");
  
  // Compile the agent
  const agent = agentBuilder.compile();
  
  // Triage router function
  const triageRouter = async (state: AgentStateType) => {
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
      
      // Run the router LLM
      const result = await llmRouter.invoke([
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ]);
      
      // Decision
      const classification = result.classification;
      
      let goto = END;
      let update: Partial<AgentStateType> = {
        classification_decision: classification,
      };
      
      if (classification === "respond") {
        console.log("📧 Classification: RESPOND - This email requires a response");
        goto = "response_agent";
        update = {
          ...update,
          messages: [
            new HumanMessage({ 
              content: `Respond to the email: ${emailMarkdown}`
            })
          ],
        };
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
    } catch (error: any) {
      console.error("Error in triage router:", error);
      // In case of error, we default to END without processing the email
      return new Command({
        goto: END,
        update: {
          classification_decision: "ignore",
          messages: [
            new SystemMessage({ 
              content: `Error processing email: ${error.message}`
            })
          ]
        }
      });
    }
  };
  
  // Define the overall workflow with properly typed nodes
  const overallWorkflow = new StateGraph<typeof AgentState.spec, 
    AgentStateType, 
    Partial<AgentStateType>,
    AgentNodes>(AgentState);
  
  overallWorkflow.addNode("triage_router", triageRouter, { ends: ["response_agent", END] });
  overallWorkflow.addNode("response_agent", agent);
  overallWorkflow.addEdge(START, "triage_router");
  
  // Compile the email assistant
  const emailAssistant = overallWorkflow.compile();
  
  return emailAssistant;
};

// For server-side usage
export async function getEmailAssistant() {
  return await createEmailAssistant();
}
