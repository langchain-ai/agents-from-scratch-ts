# Email Assistant Agents with LangGraph.js

This repository contains implementations of AI email assistants built using LangGraph.js, a library for building stateful, multi-actor applications with LLMs. It demonstrates how to create, test, and add features like Human-in-the-Loop (HITL) and persistent memory to an AI agent.

Three versions of the email assistant are available in the `src/` directory:
1.  `email_assistant.ts`: A basic email assistant for triage and response.
2.  `email_assistant_hitl.ts`: Extends the basic assistant with Human-in-the-Loop capabilities for reviewing and intervening in the agent's actions.
3.  `email_assistant_hitl_memory.ts`: Further extends the HITL assistant with persistent memory to learn from user feedback and preferences.

### Typescript Notebooks
- `langgraph_101.ipynb`
- `agent.ipynb`, 
- `hitl.ipynb`, 
- `memory.ipynb`,  


There are also multiple interactive Typescript Jupyter notebooks to go through the process of creating these agentic features. The notebooks assemble the agents in `src/` step by step. You can use **ts-lab kernel instead of the default python kernel** to execute the notebook code cells


## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Core Concepts & Workflow](#core-concepts--workflow)
  - [Overview](#overview)
  - [Key LangGraph/LangChain Components](#key-langgraphlangchain-components)
  - [General Workflow](#general-workflow)
  - [Human-in-the-Loop (HITL)](#human-in-the-loop-hitl)
  - [Memory Implementation](#memory-implementation)
- [Testing in LangGraph Studio](#testing-in-langgraph-studio)
  - [Example Email Input](#example-email-input)
  - [Understanding Node Outputs](#understanding-node-outputs)
  - [HITL Interactions & Resuming from Interrupts](#hitl-interactions--resuming-from-interrupts)
  - [Comprehensive Testing Scenarios](#comprehensive-testing-scenarios)
  - [Testing Memory Features](#testing-memory-features)
- [Development & Debugging](#development--debugging)

## Getting Started

### Prerequisites
- Install the latest version of [JupyterLab or Jupyter Notebook](https://jupyter.org/install)
- Install [TS-Lab Typescript Kernel for Notebooks](https://github.com/yunabe/tslab#display-rich-objects)
- Node.js (v18 or higher recommended)
- A package manager (npm, yarn, or pnpm)

### 1. Install Dependencies

Clone the repository and install the necessary packages:

```bash
npm install
# or
yarn install
# or
pnpm install
```

### 2. Environment Setup

Copy the `.env.example` file to  `.env` file in the root of the project and add your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```
it is recommended to also add your langsmith api key, and langsmith_project name to be able to analyze your graph traces in Langsmith.

### 3. Run the Agent

Start the email assistant. This will typically also make it available for interaction and visualization in LangGraph Studio (usually at `http://localhost:PORT/studio` - check your terminal output for the exact port).

```bash
npm run agent
# or
yarn agent
# or
pnpm agent
```

## Project Structure

- `src/`: Contains the core TypeScript source code for the email assistants.
  - `email_assistant.ts`: Basic email assistant.
  - `email_assistant_hitl.ts`: Email assistant with Human-in-the-Loop.
  - `email_assistant_hitl_memory.ts`: Email assistant with HITL and memory.
  - `tools/`: Directory for tool implementations (e.g., `base.ts`, specific tools).
  - `prompts.ts`: Contains prompt templates used by the assistants.
  - `schemas.ts`: Defines Zod schemas for state management (e.g., `BaseEmailAgentState`, `EmailAgentHITLState`) and tool inputs.
  - `utils.ts`: Utility functions for tasks like email parsing and formatting.
  - `config.ts` or `configuration.ts` (if present): For application-level configurations.
- `README.md`: This file.
- `package.json`: Project metadata and scripts.
- `tsconfig.json`: TypeScript configuration.
- `.env`: Environment variables (gitignored).
- `notebooks/` contains the interactive notebooks for the email assistants
    `0_langgraph_101.ipynb` - langgraph fundamentals
    `1_agent.ipynb` - agentic email assistant
    `2_hitl.ipynb` - Human in the Loop, Interrupts notebook
    `3_memory.ipynb` - Human in the loop, with memory notebook 

## Core Concepts & Workflow

### Overview

The email assistants are built as stateful graphs using LangGraph.js.

- **State Management**: The state of the workflow (e.g., messages, email content, classification decisions) is managed using Zod schemas defined in `src/schemas.ts` (like `BaseEmailAgentState` or `EmailAgentHITLStateType`). This provides type safety and clear structure.
- **Graph Execution**: The `StateGraph` class is used to define nodes (representing actors or functions) and edges (representing transitions based on conditions or direct flow).
- **Modularity**: Each version of the assistant (basic, HITL, memory) builds upon the previous, showcasing progressive feature integration.

### Key LangGraph/LangChain Components

- `initChatModel` (from `langchain/chat_models/universal`): Initializes the Large Language Model (e.g., GPT-4, GPT-4o).
- `StructuredTool` (from `@langchain/core/tools`): Base class for defining tools the agent can use.
- Message Types (from `@langchain/core/messages`): `BaseMessage`, `AIMessage`, `HumanMessage`, `SystemMessage`, `ToolMessage` are used for chat history and agent communication. `ToolCall` represents an LLM's request to use a tool.
- `StateGraph` (from `@langchain/langgraph`): The core class for building the graph.
- `START`, `END` (from `@langchain/langgraph`): Special nodes representing the beginning and end of a graph or subgraph.
- `Command` (from `@langchain/langgraph`): Used by nodes to direct the graph to the next node and update the state.
- `interrupt` (from `@langchain/langgraph`): Pauses graph execution for Human-in-the-Loop interactions.

### General Workflow

1.  **Email Input & Parsing**:

    - The assistant receives an email (structured as defined in `src/schemas.ts`).
    - `parseEmail` (from `src/utils.ts`) extracts key information (author, recipients, subject, body/thread).
    - `formatEmailMarkdown` (from `src/utils.ts`) prepares the email content for the LLM.

2.  **Triage (`triage_router` node)**:

    - An LLM call classifies the email into one of three categories:
      - `respond`: Requires a direct response.
      - `notify`: Contains important information but may not need a direct response (can be reviewed by a human in HITL versions).
      - `ignore`: Can be safely filtered out.
    - The graph transitions based on this classification.

3.  **Response Generation (`response_agent` subgraph)**:
    - If an email is classified as `respond` (or a `notify` email is escalated), this subgraph is invoked.
    - **LLM Calls (`llm_call` node)**: The LLM, equipped with tools, decides the next action (e.g., draft a reply, use a tool to find information, or finish).
    - **Tool Usage**:
      - In `email_assistant.ts`, a `ToolNode` (named `environment`) executes tool calls.
      - In HITL versions, tool execution is often integrated within the `interruptHandlerNode` after potential human review.
    - **Conditional Logic (`shouldContinue` function)**: Determines if the agent needs to continue (e.g., make more tool calls) or if it has completed its task.

### Human-in-the-Loop (HITL)

Implemented in `email_assistant_hitl.ts` and `email_assistant_hitl_memory.ts`.
Interactive Notebooks available in `2_hitl.ipynb` and `3_memory.ipynb`
- **Interrupts**: The graph execution pauses at critical junctures:
  - After triage if an email is marked `notify` (`triage_interrupt_handler` node).
  - Before executing certain tool calls within the `response_agent` (`interrupt_handler` node).
- **Human Review**: Users can:
  - **Review** proposed actions (e.g., email drafts, tool parameters).
  - **Edit** data (e.g., modify an email draft before sending).
  - **Accept** the agent's proposed action.
  - **Ignore/Reject** the action.
  - Provide **feedback** to guide the agent's next step.

### Memory Implementation

Implemented in `email_assistant_hitl_memory.ts`.

- **Persistent Learning**: The assistant learns from user interactions and feedback to improve its performance over time.
- **Memory Storage**: Uses `InMemoryStore` (can be swapped for other persistent stores).
- **Key Functions**:
  - `getMemory`: Retrieves preferences (e.g., for triage, response style) from the store or initializes with defaults.
  - `updateMemory`: Intelligently updates memory profiles based on user feedback or edits during HITL interactions.
- **Namespaces**: Memory is organized into distinct namespaces for different types of preferences:
  - `["email_assistant", "triage_preferences"]`: Rules for email classification.
  - `["email_assistant", "response_preferences"]`: Preferences for email writing style.
  - `["email_assistant", "cal_preferences"]`: Preferences related to calendar scheduling.

## Testing in LangGraph Studio

LangGraph Studio provides a visual interface to run, debug, and interact with your email assistant graph.

### Example Email Input

Use this JSON structure as input when testing your agent in the studio:

```json
{
  "email_input": {
    "id": "email_123456789",
    "thread_id": "thread_abc123",
    "from_email": "client@example.com",
    "to_email": "support@yourcompany.com",
    "subject": "Request for Meeting Next Tuesday",
    "page_content": "Hi Support Team,\n\nI hope this email finds you well. I'd like to schedule a meeting next Tuesday to discuss our ongoing project implementation. We have some questions about the timeline and deliverables that would be best addressed in a call.\n\nWould 2:00 PM EST work for you? If not, please suggest a time that works with your schedule.\n\nBest regards,\nJohn Smith\nProject Manager\nClient Company Inc.",
    "send_time": "2025-05-01T10:30:00Z"
  }
}
```

### Understanding Node Outputs

#### Triage Router Node (`triage_router`)
Classifies emails. Expect output like:

```json
{
  "classification_decision": "respond", // or "notify", "ignore"
  "messages": [
    // Updated messages array in the state
    {
      "content": "Respond to the email: ...", // Example human message added
      "type": "human"
    }
  ]
}
```

#### Response Agent Nodes (e.g., `llm_call`, `interrupt_handler`)
Handles response generation.

- `llm_call` output might include tool calls:

```json
{
  "messages": [
    // Could be a single AIMessage with tool_calls
    {
      "content": null,
      "tool_calls": [
        {
          "name": "write_email",
          "args": {
            "recipient": "client@example.com",
            "subject": "RE: Request for Meeting Next Tuesday",
            "body": "Hi John,\n\nThanks for reaching out! Yes, 2:00 PM EST next Tuesday works for our team. I've added it to our calendar.\n\nLooking forward to discussing the project timeline and deliverables with you.\n\nBest regards,\nSupport Team"
          }
        }
      ],
      "type": "ai"
    }
  ]
}
```

- `interrupt_handler` output will include tool responses after execution or human feedback.

### HITL Interactions & Resuming from Interrupts

When an interrupt pauses the graph for human input, you resume by providing an array containing a single `HumanResponse` object. The structure depends on the action:

**1. Accepting an Action** (e.g., `allow_accept: true`)

```json
[
  {
    "type": "accept"
  }
]
```

**2. Editing Arguments** (e.g., modifying a draft email for `write_email` tool)

```json
[
  {
    "type": "edit",
    "args": {
      // Structure of 'args' must match the tool's expected input
      "recipient": "client@example.com",
      "subject": "Re: Your Updated Question",
      "body": "Hello Sarah,\n\nThanks for the clarification! I have now updated the information regarding the '/users' endpoint. Please find the revised details attached.\n\nBest regards,\nLance\nSupport Team"
    }
  }
]
```

**3. Providing Feedback/Response** (e.g., giving instructions to the LLM)

```json
[
  {
    "type": "response",
    "args": "The draft is good, but please make the tone slightly more formal and add a closing sentence about looking forward to their reply."
  }
]
```

**4. Ignoring an Action** (e.g., `allow_ignore: true`)

```json
[
  {
    "type": "ignore"
  }
]
```

### Comprehensive Testing Scenarios

To thoroughly test all graph paths:

#### 1. Email Requiring Response (Triage → Response Agent)

**Input:**

```json
{
  "email_input": {
    "id": "email_123",
    "thread_id": "thread_abc123",
    "from_email": "client@example.com",
    "to_email": "support@yourcompany.com",
    "subject": "Question about API documentation",
    "page_content": "Hello Support Team,\nI'm working on integrating with your API and I can't find the documentation for the '/users' endpoint. Could you please point me to where I can find this information or provide details about the request and response formats?\nThank you,\nSarah Johnson",
    "send_time": "2025-05-01T09:00:00Z"
  }
}
```

**Expected:** `triage_router` classifies as `respond`, then `response_agent` generates a reply (possibly using tools).

#### 2. Notification Email (Triage → `triage_interrupt_handler` [for HITL versions])

**Input:**

```json
{
  "email_input": {
    "id": "email_456",
    "thread_id": "thread_def456",
    "from_email": "system@example.com",
    "to_email": "all@yourcompany.com",
    "subject": "System Maintenance - Important Notice",
    "page_content": "Dear Team,\nThis is to inform you that we will be conducting scheduled system maintenance this Saturday from 10:00 PM to 2:00 AM EST. During this time, the production server will be temporarily unavailable.\nRegards,\nIT Operations",
    "send_time": "2025-05-02T12:00:00Z"
  }
}
```

**Expected (HITL):** `triage_router` classifies as `notify`. `triage_interrupt_handler` pauses for human review.

- **User chooses "respond" (with feedback):** Flow proceeds to `response_agent`.
- **User chooses "ignore":** Flow proceeds to `END`.

#### 3. Email to Ignore (Triage → END)

**Input:**

```json
{
  "email_input": {
    "id": "email_789",
    "thread_id": "thread_ghi789",
    "from_email": "marketing@newsletter.com",
    "to_email": "user@yourcompany.com",
    "subject": "50% Off Spring Sale - Limited Time Offer!",
    "page_content": "AMAZING DEALS JUST FOR YOU! Don't miss our Spring Sale...",
    "send_time": "2025-05-03T08:00:00Z"
  }
}
```

**Expected:** `triage_router` classifies as `ignore`. Flow proceeds to `END`.

#### 4. Testing Response Agent Tool Call Interrupts (HITL versions)

When the `response_agent` (via `interrupt_handler`) proposes a tool call (e.g., `write_email`, `schedule_meeting`):

- **Accept:** Tool executes with original arguments.
- **Edit:** Modify arguments (e.g., email body) before execution.
- **Ignore:** Tool call is rejected, potentially ending the workflow or prompting LLM for alternative.
- **Feedback:** Send instructions back to the LLM (e.g., "Make the tone more conversational").

### Testing Memory Features (`email_assistant_hitl_memory.ts`)

1.  Run sequences of emails with similar themes.
2.  During HITL, consistently edit responses or provide feedback (e.g., always making emails more concise, or always changing meeting suggestions to afternoons).
3.  After several interactions, send a new email of the same type. Observe if the agent's initial suggestions (triage classification, draft content, meeting times) adapt based on the learned preferences stored in memory.
    - Example: If you consistently ignore system maintenance emails after they are classified as `notify`, the `triage_preferences` might update to classify them as `ignore` directly.
