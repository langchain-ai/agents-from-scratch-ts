# Gmail Integration Tools

This directory contains tools for integrating with Gmail and Google Calendar APIs to enable the email assistant to work with real emails and calendar events.

## Features

- **Email Fetching**: Retrieve recent emails from your Gmail account
- **Email Sending**: Send replies to email threads
- **Calendar Availability**: Check your Google Calendar for availability on specific dates
- **Meeting Scheduling**: Create calendar events and send invites to attendees

## Setup Instructions

### 1. Set up Google Cloud Project and Enable Gmail API

1. Enable the Gmail API by clicking the blue "Enable API" button [here](https://developers.google.com/gmail/api/quickstart/python#enable_the_api)

2. Authorize credentials for a desktop application [here](https://developers.google.com/workspace/gmail/api/quickstart/python#authorize_credentials_for_a_desktop_application)
- Go to Clients 
- Create Client
- Application type > Desktop app
- Create
- Under "Audience" select "External" if you're using a personal email (non-Google Workspace)

<img width="1496" alt="Screenshot 2025-04-26 at 7 43 57 AM" src="https://github.com/user-attachments/assets/718da39e-9b10-4a2a-905c-eda87c1c1126" />

- Add yourself as a test user

<img width="1622" alt="Screenshot 2025-04-26 at 7 46 32 AM" src="https://github.com/user-attachments/assets/0489ad7e-0acd-4abd-b309-7c97ce705932" />

3. Save the downloaded JSON file

### 3. Set Up Authentication Files

1. Move your downloaded client secret JSON file to the `.secrets` directory

```bash
# Create a secrets directory
mkdir -p src/email_assistant/tools/gmail/.secrets

# Move your downloaded client secret to the secrets directory
mv /path/to/downloaded/client_secret.json src/email_assistant/tools/gmail/.secrets/secrets.json
```

2. Run the Gmail setup script

```bash
# Run the Gmail setup script
python src/email_assistant/tools/gmail/setup_gmail.py
```

-  This will open a browser window for you to authenticate with your Google account
-  This will create a `token.json` file in the `.secrets` directory
-  This token will be used for Gmail API access

### 4. Run the Gmail Ingestion Script

1. Once you have authentication set up, you can run the Gmail ingestion script. 

2. Start the locally running LangGraph server in one terminal:

```
langgraph dev
```

3. Run the ingestion script in another terminal:

```bash
python src/email_assistant/tools/gmail/run_ingest.py --email rlance.martin@gmail.com --minutes-since 1000
```

- This will fetch emails from the past 1000 minutes and process them with your email assistant.
- It will use the LangGraph SDK to pass each email to the locally running email assistant.

#### Important Ingestion Parameters:

- `--graph-name`: Name of the LangGraph to use (default: "email_assistant_hitl_memory")
- `--email`: The email address to fetch messages from (alternative to setting EMAIL_ADDRESS)
- `--minutes-since`: Only process emails that are newer than this many minutes (default: 60)
- `--url`: URL of the LangGraph deployment (default: http://127.0.0.1:2024)
- `--rerun`: Process emails that have already been processed (default: false)
- `--early`: Stop after processing one email (default: false)
- `--mock`: Run in mock mode without requiring a LangGraph server
- `--include-read`: Include emails that have already been read (by default only unread emails are processed)
- `--skip-filters`: Process all emails without filtering (by default only latest messages in threads where you're not the sender are processed)
- `--enable-tracing`: Enable LangSmith tracing (requires LANGCHAIN_API_KEY to be set)
- `--langsmith-api-key`: LangSmith API key for tracing (alternative to setting LANGCHAIN_API_KEY)
- `--langsmith-project`: LangSmith project name for tracing (default: "gmail-assistant")

#### Flag Combinations:

- `--rerun --early`: Process one email (regardless if it was processed before) and stop
- `--rerun`: Process all emails, including ones previously processed by LangGraph
- `--early`: Process only one new (previously unprocessed) email and stop
- (no flags): Process only new (previously unprocessed) emails
- `--include-read --skip-filters`: Process all emails, including ones marked as read and ones that would normally be filtered out
- `--minutes-since 1000 --include-read --skip-filters`: Process all emails from the past ~16 hours without any filtering
- `--enable-tracing --langsmith-project "my-project"`: Process emails with LangSmith tracing enabled

#### Troubleshooting:

- **Missing emails?** The Gmail API applies filters to show only important/primary emails by default. You can:
  - Increase the `--minutes-since` parameter to a larger value (e.g., 1000) to fetch emails from a longer time period
  - Use the `--include-read` flag to process emails marked as "read" (by default only unread emails are processed)
  - Use the `--skip-filters` flag to include all messages (not just the latest in a thread, and including ones you sent)
  - Try running with all options to process everything: `--include-read --skip-filters --minutes-since 1000`
  - Use the `--mock` flag to test the system with simulated emails

## How Gmail Ingestion Works

The Gmail ingestion process works in three main stages:

### 1. CLI Parameters → Gmail Search Query

CLI parameters are translated into a Gmail search query:

- `--minutes-since 1440` → `after:TIMESTAMP` (emails from the last 24 hours)
- `--email you@example.com` → `to:you@example.com OR from:you@example.com` (emails where you're sender or recipient)
- `--include-read` → removes `is:unread` filter (includes read messages)

For example, running:
```
python run_ingest.py --email you@example.com --minutes-since 1440 --include-read
```

Creates a Gmail API search query like:
```
(to:you@example.com OR from:you@example.com) after:1745432245
```

### 2. Search Results → Thread Processing

For each message returned by the search:

1. The script obtains the thread ID
2. Using this thread ID, it fetches the **complete thread** with all messages
3. Messages in the thread are sorted by date to identify the latest message
4. Depending on filtering options, it processes either:
   - The specific message found in the search (default behavior)
   - The latest message in the thread (when using `--skip-filters`)

### 3. Default Filters and `--skip-filters` Behavior

#### Default Filters Applied

Without `--skip-filters`, the system applies these three filters in sequence:

1. **Unread Filter** (controlled by `--include-read`):
   - Default behavior: Only processes unread messages 
   - With `--include-read`: Processes both read and unread messages
   - Implementation: Adds `is:unread` to the Gmail search query
   - This filter happens at the search level before any messages are retrieved

2. **Sender Filter**:
   - Default behavior: Skips messages sent by your own email address
   - Implementation: Checks if your email appears in the "From" header
   - Logic: `is_from_user = email_address in from_header`
   - This prevents the assistant from responding to your own emails

3. **Thread-Position Filter**:
   - Default behavior: Only processes the most recent message in each thread
   - Implementation: Compares message ID with the last message in thread
   - Logic: `is_latest_in_thread = message["id"] == last_message["id"]`
   - Prevents processing older messages when a newer reply exists
   
The combination of these filters means only the latest message in each thread that was not sent by you and is unread (unless `--include-read` is specified) will be processed.

#### Effect of `--skip-filters` Flag

When `--skip-filters` is enabled:

1. **Bypasses Sender and Thread-Position Filters**:
   - Messages sent by you will be processed
   - Messages that aren't the latest in thread will be processed
   - Logic: `should_process = skip_filters or (not is_from_user and is_latest_in_thread)`

2. **Changes Which Message Is Processed**:
   - Without `--skip-filters`: Uses the specific message found by search
   - With `--skip-filters`: Always uses the latest message in the thread
   - Even if the latest message wasn't found in the search results

3. **Unread Filter Still Applies (unless overridden)**:
   - `--skip-filters` does NOT bypass the unread filter
   - To process read messages, you must still use `--include-read`
   - This is because the unread filter happens at the search level

In summary:
- Default: Process only unread messages where you're not the sender and that are the latest in their thread
- `--skip-filters`: Process all messages found by search, using the latest message in each thread
- `--include-read`: Include read messages in the search
- `--include-read --skip-filters`: Most comprehensive, processes the latest message in all threads found by search

## Important Gmail API Limitations

The Gmail API has several limitations that affect email ingestion:

1. **Search-Based API**: Gmail doesn't provide a direct "get all emails from timeframe" endpoint
   - All email retrieval relies on Gmail's search functionality
   - Search results can be delayed for very recent messages (indexing lag)
   - Search results might not include all messages that technically match criteria

2. **Two-Stage Retrieval Process**:
   - Initial search to find relevant message IDs
   - Secondary thread retrieval to get complete conversations
   - This two-stage process is necessary because search doesn't guarantee complete thread information

## When to Use `--skip-filters`

### Use `--skip-filters` When:

- **Latest Messages Are Missing**: The thread contains newer messages that aren't being processed
- **Complete Thread Context Needed**: You want to ensure you have the most up-to-date conversation context
- **Debugging Thread Issues**: You need to see which messages exist in threads vs. which are being processed
- **Initial Data Loading**: You're populating the system with existing conversations
- **Inconsistent Results**: You notice some messages are being skipped or processed out of order

### When NOT to Use `--skip-filters`:

- **Day-to-Day Operation**: For routine email processing, the default filters provide a natural workflow
- **Avoiding Duplicates**: To prevent reprocessing messages that have already been handled
- **Targeting Specific Messages**: When you want to process exactly the messages that match your search criteria
- **Processing Only New Correspondence**: When you want to handle only new, unread messages directed to you

### Behavior With `--skip-filters` Enabled:

1. The system still uses search to find relevant thread IDs
2. For each thread found, it fetches ALL messages in that thread
3. It sorts all messages by timestamp to identify the truly latest message
4. It processes the latest message in each thread, even if:
   - That message wasn't in the original search results
   - That message was sent by you
   - That message isn't the latest in the original search results

This ensures you're always working with the most current state of each conversation.

## Known Limitations and Troubleshooting

- **Indexing Delays**: The Gmail API's search might miss very recent messages (added in the last few minutes)
- **Inconsistent Threading**: Gmail's thread IDs are consistent within a session but might change across API calls
- **Message Visibility**: Some messages might be excluded due to Gmail's categorization (Promotions, Updates, etc.)
- **Rate Limits**: The Gmail API has rate limits that could affect processing of large email volumes

If messages appear to be missing:
- Use a larger `--minutes-since` value to cast a wider time net
- Enable `--include-read` to include messages you've already read
- Enable `--skip-filters` to process the latest message in each thread
- Try the combination: `--minutes-since 1440 --include-read --skip-filters`

- **Connection errors:** If you get "Connection refused" or "All connection attempts failed" errors:
  - Make sure the LangGraph server is running with `langgraph start` in a separate terminal
  - Verify the port number matches in your script (default is 2024)
  - Use the `--mock` flag to test without a LangGraph server: `--mock`

## Recent Updates and Fixes

The Gmail integration has been updated with several improvements:

1. **Improved Thread Processing**: Now properly retrieves all messages in a thread, not just the ones found by search
   - Added comprehensive logging of thread messages with dates and senders
   - Fixed sorting to ensure the truly latest message is identified

2. **Enhanced `--skip-filters` Behavior**: When enabled, the system now:
   - Processes the absolute latest message in the thread, even if it wasn't found in search
   - Uses thread-based retrieval to bypass Gmail search limitations
   - Shows detailed information about which messages are being processed

3. **Thread ID Handling**: Improved how thread IDs are mapped between Gmail and LangGraph
   - Uses MD5 hash to ensure consistent ID generation across runs
   - Better error handling for thread ID mapping issues

4. **Simplified Command-Line Interface**: 
   - Improved flag handling with boolean flags for better usability
   - Added LangSmith tracing options for better observability
   - Simplified parameters and added clearer documentation

5. **LangSmith Tracing Integration**: 
   - Added support for tracing email processing through LangSmith
   - Ensured tracing context is maintained across workflow interrupts
   - Added explicit flags for controlling tracing behavior

- **Authentication issues:** If you encounter a "Token has been expired or revoked" error, delete the existing `token.json` file and run the setup script again to generate a fresh token.
- **Tracing issues:** If you're not seeing traces in LangSmith after interrupts, ensure you're using the latest version of LangSmith.

## Using Gmail Tools in Your Agent

To use Gmail tools in your agent, modify your agent code as follows:

```python
from src.email_assistant.tools import get_tools, get_tools_by_name
from src.email_assistant.tools.gmail.prompt_templates import COMBINED_TOOLS_PROMPT

# Get tools with Gmail integration enabled
tools = get_tools(include_gmail=True)
tools_by_name = get_tools_by_name(tools)

# Use the combined tools prompt in your agent's system prompt
system_prompt = agent_system_prompt.format(
    tools_prompt=COMBINED_TOOLS_PROMPT,
    # other parameters...
)
```

See `src/email_assistant/gmail_assistant.py` for a complete example.
