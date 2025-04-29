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
import { BaseMessage, HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import { ToolCall, ToolMessage } from "@langchain/core/messages/tool";
import { isToolMessage } from "@langchain/core/messages/tool";

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
  
  // Nodes
  const llmCall = async (state: typeof AgentState.State) => {
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
  
  const toolNode = async (state: typeof AgentState.State) => {
    /**
     * Performs the tool call based on LLM's decision
     * Similar to the Python version's tool_node function
     */
    const messages = [...state.messages];
    const lastMessage = messages[messages.length - 1];
    const result: BaseMessage[] = [];
    
    // Check if this message has tool calls
    if (lastMessage instanceof AIMessage && lastMessage.tool_calls && Array.isArray(lastMessage.tool_calls)) {
      for (const toolCall of lastMessage.tool_calls) {
        try {
          const tool = toolsByName[toolCall.name];
          if (!tool) {
            throw new Error(`Tool not found: ${toolCall.name}`);
          }
          const observation = await tool.invoke(toolCall.args);
          
          // Create a proper tool message
          const toolMessage = new ToolMessage({
            content: observation,
            tool_call_id: toolCall.id ?? ""
          });
          
          result.push(toolMessage);
        } catch (error: any) {
          console.error(`Error executing tool ${toolCall.name}:`, error);
          // Create error message
          const errorMessage = new ToolMessage({
            content: `Error executing tool ${toolCall.name}: ${error.message}`,
            tool_call_id: toolCall.id ?? ""
          });
          
          result.push(errorMessage);
        }
      }
    }
    
    return {
      messages: result
    };
  };
  
  // Conditional edge function
  const shouldContinue = (state: typeof AgentState.State) => {
    /**
     * Route to environment for tool execution, or end if Done tool called
     * Similar to the Python version's should_continue function
     */
    const messages = state.messages;
    if (!messages || messages.length === 0) return END;
    
    const lastMessage = messages[messages.length - 1];
    
    if (lastMessage instanceof AIMessage && lastMessage.tool_calls && Array.isArray(lastMessage.tool_calls)) {
      for (const toolCall of lastMessage.tool_calls) {
        if (toolCall.name === "Done") {
          return END;
        }
      }
      return "environment";
    }
    
    return END;
  };
  
  // Define node names as a union type
  type AgentNodes = typeof START | typeof END | "llm_call" | "environment" | "triage_router" | "response_agent";

  // Use the type parameter to specify allowed node names
  const agentBuilder = new StateGraph<typeof AgentState.spec, 
    typeof AgentState.State, 
    Partial<typeof AgentState.State>,
    AgentNodes>(AgentState);
  
  // Add nodes
  agentBuilder.addNode("llm_call", llmCall);
  agentBuilder.addNode("environment", toolNode);
  
  // Add edges to connect nodes
  agentBuilder.addEdge(START, "llm_call");
  agentBuilder.addConditionalEdges("llm_call", shouldContinue);
  agentBuilder.addEdge("environment", "llm_call");
  
  // Compile the agent
  const agent = agentBuilder.compile();
  
  // Triage router function
  const triageRouter = async (state: typeof AgentState.State) => {
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
      let update: Partial<typeof AgentState.State> = {
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
  
  // Input state definition
  const InputState = Annotation.Root({
    email_input: Annotation<EmailData>(),
  });
  
  // Similar approach for overallWorkflow

  const overallWorkflow = new StateGraph<typeof AgentState.spec, 
  typeof AgentState.State, 
  Partial<typeof AgentState.State>,
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
