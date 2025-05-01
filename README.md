## Python Translation To-Do
- [x] prompts 
- [x] schemas  
- [x] mock tools 
- [x] utils 
- [x] email assistant 
- [Th APR 31 ] email_assistant_hitl
- [ Fri May 1] email_assistant_memory

## TypeScript Email Assistant Implementation

This is a vanilla TypeScript project for email assistant workflows using LangChain and LangGraph.

### Architecture
- Uses `StateGraph` from LangGraph to create a multi-step workflow
- Leverages `Annotation` to track state across the graph nodes
- Two main components: triage router + email response agent

### Key LangChain/LangGraph Components:
- `initChatModel`: Creates LLM instance
- `StructuredTool`: Base class for tool definitions
- `BaseMessage`, `AIMessage`, `HumanMessage`, `SystemMessage`: Message handling
- `ToolMessage`, `ToolCall`: Tool interaction types
- `messagesStateReducer`: Manages message state history
- `StateGraph`: Main orchestration framework
- `Command`: Directs state transitions in the graph

### Workflow Sequence:
1. `triageRouter`: Classifies email as respond/ignore/notify
2. If respond → `response_agent` (compiled agent workflow)
3. Agent workflow:
   - `llmCall`: Makes decisions using LLM + bound tools
   - `toolNode`: Executes tool calls with error handling
   - `shouldContinue`: Determines if more tool calls needed

### Email Processing:
- Parses emails with `parseEmail` → author, to, subject, thread
- Formats email content with `formatEmailMarkdown`
- Routes to appropriate handling based on classification

### State Management:
- `AgentState` tracks messages, email input, and classification
- Properly typed with TypeScript for complete type safety
- Uses command pattern to transition between states

### Project Structure
- `scripts/`: TypeScript scripts to run the email assistant
- `lib/`: Utility functions, tools, and shared types
  - `lib/tools/`: Tool implementations
  - `lib/prompts.ts`: Prompt templates
  - `lib/schemas.ts`: TypeScript/Zod schemas
  - `lib/utils.ts`: Utility functions

## Getting Started

First, install dependencies:

```bash
npm install
# or
yarn install
# or
pnpm install
```

Then, create a `.env` file in the root of the project with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

Run the standard email assistant:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Run the human-in-the-loop (HITL) email assistant:

```bash
npm run dev:hitl
# or
yarn dev:hitl
# or
pnpm dev:hitl
```

## Course outline 
> BUILD

> EVAL

> HITL

> MEMORY

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```



