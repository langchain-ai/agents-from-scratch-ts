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

### Testing All Graph Nodes

To thoroughly test the entire graph and all possible paths, use the following examples:

#### 1. Email Requiring Response (Triage → Response Agent)

```json
{
  "email_input": {
    "id": "email_123456789",
    "thread_id": "thread_abc123",
    "from_email": "client@example.com",
    "to_email": "support@yourcompany.com",
    "subject": "Question about API documentation",
    "page_content": "Hello Support Team,\n\nI'm working on integrating with your API and I can't find the documentation for the '/users' endpoint. Could you please point me to where I can find this information or provide details about the request and response formats?\n\nThank you,\nSarah Johnson\nDeveloper\nClient Tech Inc.",
    "send_time": "2025-05-01T14:22:00Z"
  }
}
```

#### 2. Notification Email (Triage → Triage Interrupt Handler)

```json
{
  "email_input": {
    "id": "email_789012345",
    "thread_id": "thread_def456",
    "from_email": "system@example.com",
    "to_email": "team@yourcompany.com",
    "subject": "System Maintenance - Important Notice",
    "page_content": "Dear Team,\n\nThis is to inform you that we will be conducting scheduled system maintenance this Saturday from 10:00 PM to 2:00 AM EST. During this time, the production server will be temporarily unavailable.\n\nPlease plan your work accordingly and ensure any critical tasks are completed before this maintenance window.\n\nRegards,\nIT Operations",
    "send_time": "2025-05-02T09:15:00Z"
  }
}
```

#### 3. Email to Ignore (Triage → END)

```json
{
  "email_input": {
    "id": "email_345678901",
    "thread_id": "thread_ghi789",
    "from_email": "marketing@newsletter.com",
    "to_email": "all-staff@yourcompany.com",
    "subject": "50% Off Spring Sale - Limited Time Offer!",
    "page_content": "AMAZING DEALS JUST FOR YOU!\n\nDon't miss our Spring Sale with 50% off all products!\n\nUse promo code SPRING50 at checkout.\nOffer valid until May 15th.\n\nShop Now! Click here to browse our collection.\n\nTo unsubscribe from our mailing list, click here.",
    "send_time": "2025-05-03T11:30:00Z"
  }
}
```

#### 4. Testing Triage Interrupt → Response Agent Path

When the triage node classifies an email as "notify", you'll be prompted with an interrupt. To test this path:

1. Use the "Notification Email" example above
2. When the interrupt appears, choose the "continue" or "feedback" action
3. Optionally provide feedback like: "Please respond to this with our availability during the maintenance window"
4. This will direct the flow to the response_agent node

#### 5. Testing Triage Interrupt → END Path

When the triage node classifies an email as "notify", you can also choose to ignore it:

1. Use the "Notification Email" example above
2. When the interrupt appears, choose the "ignore" action
3. This will end the workflow

#### 6. Testing Response Agent Tool Calls

When the response agent makes a tool call, you'll be prompted with an interrupt. To test different paths:

**Accept the tool call:**
Choose "accept" to execute the tool with the original arguments

**Edit the tool call:**
Choose "edit" to modify the tool arguments before execution. For example, if the agent drafts an email, you might modify the wording or add additional information.

**Ignore the tool call:**
Choose "ignore" to reject the tool call, which will send the workflow to END

**Provide feedback:**
Choose "feedback" to send instructions back to the agent. For example: "The tone is too formal, please make it more conversational"

### Resuming from Interrupts: Input Examples

When an interrupt pauses the graph execution for human input, you'll resume the graph by providing an array containing a single `HumanResponse` object. The structure of this object depends on the action you choose (accept, edit, ignore, or respond).

Here are examples for common scenarios:

**1. Accepting an Action (e.g., a tool call)**

If the interrupt allows accepting (e.g., `allow_accept: true`), and you choose to accept the proposed action without changes:

```json
[
  {
    "type": "accept"
  }
]
```

**2. Editing Arguments (e.g., modifying a draft email)**

If the interrupt allows editing (e.g., `allow_edit: true`), you provide the new arguments for the action. For example, if editing a `write_email` tool call:

```json
[
  {
    "type": "edit",
    "args": {
      "recipient": "client@example.com",
      "subject": "Re: Your Updated Question",
      "content": "Hello Sarah,\n\nThanks for the clarification! I have now updated the information regarding the '/users' endpoint. Please find the revised details attached.\n\nBest regards,\nLance\nSupport Team"
    }
  }
]
```

_Note: The structure of `args` must match what the interrupted tool or action expects._

**3. Providing Feedback/Response (e.g., giving instructions to the LLM)**

If the interrupt allows responding (e.g., `allow_respond: true`), you provide a string as the `args`:

```json
[
  {
    "type": "response",
    "args": "The draft is good, but please make the tone slightly more formal and add a closing sentence about looking forward to their reply."
  }
]
```

**4. Ignoring an Action**

If the interrupt allows ignoring (e.g., `allow_ignore: true`), and you choose to ignore the proposed action:

```json
[
  {
    "type": "ignore"
  }
]
```

These examples cover the primary ways you'll interact with interrupts in LangGraph Studio. The specific `args` needed for the "edit" type will depend on the tool or action that was interrupted.

### Testing Memory Features

After running a few interactions with different types of emails, the memory features should begin to adapt. Try the following:

1. Run a sequence of emails with similar themes
2. Edit several email responses in a consistent way (e.g., always making them more concise)
3. Then send a new email of the same type and observe if the agent adapts its behavior

For example, if you consistently ignore system maintenance emails, eventually the triage preferences should update to classify similar emails as "ignore" instead of "notify".

### Testing Different Scenarios

Try modifying the email content to test different classification outcomes:

1. **For "respond" classification**: Include direct questions, meeting requests, or explicit asks
2. **For "notify" classification**: Send FYI updates without questions or required actions
3. **For "ignore" classification**: Use marketing language, irrelevant information, or messages clearly meant for others

### Memory Implementation

#### Memory Management:

`getMemory:` Retrieves memory from the store or initializes with defaults
`updateMemory:` Intelligently updates memory based on user feedback

#### Memory is organized in namespaces for different aspects:

- ["email_assistant", "triage_preferences"]: Email classification preferences
- ["email_assistant", "response_preferences"]: Email response style preferences
- ["email_assistant", "cal_preferences"]: Calendar meeting preferences

## TS Video outline

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
- [ ] graph diagrams `await graph.getGraph().drawMermaidPng()`
