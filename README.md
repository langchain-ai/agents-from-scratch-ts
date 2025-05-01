## Getting Started

1. Install dependencies:

```bash
npm install
# or
yarn install
# or
pnpm install
```

2.  create a `.env` file in the root of the project with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```


3. Start Langgraph Studio 

```bash
npm run agent
# or
yarn agent
# or
pnpm agent
```

## Table of Contents

- [TypeScript Email Assistant Implementation](#typescript-email-assistant-implementation)
- [Project Structure](#project-structure)
- [LangGraph Studio Testing](#langgraph-studio-testing)
  - [Node Details and Expected Outputs](#node-details-and-expected-outputs)
  - [Testing Different Scenarios](#testing-different-scenarios)
- [Course Outline](#course-outline)
- [To-Do List](#to-do-list-tues-apr-29---fri-may-2nd)


### Project Structure
- `scripts/`: TypeScript scripts to run the email assistant
- `lib/`: Utility functions, tools, and shared types
  - `lib/tools/`: Tool implementations
  - `lib/prompts.ts`: Prompt templates
  - `lib/schemas.ts`: TypeScript/Zod schemas
  - `lib/utils.ts`: Utility functions

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




## LangGraph Studio Testing


To test the email assistant in LangGraph Studio, use this example email input:
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

### Node Details and Expected Outputs

#### Triage Router Node
This node analyzes the email content and classifies it into one of three categories:

1. **respond** - Emails that require a direct response, such as:
   - Specific questions from clients or teammates
   - Meeting requests and scheduling communications
   - Task assignments directed to you
   - Direct inquiries about projects, timelines, or deliverables

2. **notify** - Important information that doesn't need a direct response:
   - FYI emails and project updates
   - Announcements that are relevant to your work
   - Information that should be noted but doesn't require action
   - Status updates from team members

3. **ignore** - Emails that can be safely filtered out:
   - Marketing and promotional emails
   - Irrelevant company-wide announcements
   - Emails clearly meant for other teams or departments
   - Automated system notifications

Output example after classification:
```json
{
  "classification_decision": "respond",
  "messages": [
    {
      "content": "Respond to the email: ...",
      "type": "human"
    }
  ]
}
```

#### Response Agent Node
This node handles the actual email response generation. It can:

1. Use tools to craft appropriate responses
2. Look up information if needed
3. Check calendars for meeting availability
4. End the process when a satisfactory response is formulated

Tool call output example:
```json
{
  "messages": [
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

#### Human-in-the-Loop Interactions
For the HITL version, interrupts occur at decision points, allowing you to:

1. **Review** actions before they're executed
2. **Modify** content (like email drafts) before sending
3. **Provide feedback** to guide the assistant
4. **Approve or reject** proposed actions

The interrupt dialog will show the proposed action and expected outcome, allowing you to make informed decisions during the workflow execution.

### Testing Different Scenarios

Try modifying the email content to test different classification outcomes:

1. **For "respond" classification**: Include direct questions, meeting requests, or explicit asks
2. **For "notify" classification**: Send FYI updates without questions or required actions
3. **For "ignore" classification**: Use marketing language, irrelevant information, or messages clearly meant for others


## Course outline 
> BUILD

> EVAL

> HITL

> MEMORY

## To Do List Tues Apr 29 - Fri May 2nd
- [x] prompts 
- [x] schemas  
- [x] mock tools 
- [x] utils 
- [x] email assistant 
- [x ] email_assistant_hitl
- [ in progress] email_assistant_memory
- [ ] improve implementations
- [ ] improve structure to mirror Python more closely 
- [ ] graph diagrams ```await graph.getGraph().drawMermaidPng()```

