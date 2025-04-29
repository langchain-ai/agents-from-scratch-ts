"use client";
// this is a temporary page for testing the email assistant (before setting up LangSmith)
import { useState } from "react";
import { EmailData } from "../../lib/schemas";
import { BaseMessage } from "@langchain/core/messages";

export default function TestEmailAssistant() {
  const [emailInput, setEmailInput] = useState<string>(
    `From: john@example.com
To: assistant@company.com
Subject: Meeting Request

Hi there,

Can we schedule a meeting for tomorrow at 2pm to discuss the project?

Thanks,
John`
  );
  
  const [output, setOutput] = useState<any>({});
  const [loading, setLoading] = useState<boolean>(false);
  const [messages, setMessages] = useState<BaseMessage[]>([]);
  
  async function runTest() {
    setLoading(true);
    try {
      // Call the API route instead of directly invoking the assistant
      const response = await fetch('/api/email-assistant', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email_input: emailInput
        }),
      });
      
      if (!response.ok) {
        throw new Error(`API responded with status: ${response.status}`);
      }
      
      const result = await response.json();
      
      setOutput(result);
      
      // Extract messages if they exist
      if (result.messages && Array.isArray(result.messages)) {
        setMessages(result.messages);
      }
    } catch (error: any) {
      console.error("Error running email assistant:", error);
      setOutput({ error: error.message });
    } finally {
      setLoading(false);
    }
  }
  
  return (
    <div className="container mx-auto p-4 max-w-4xl">
      <h1 className="text-2xl font-bold mb-4">Email Assistant Test</h1>
      
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Email Input</label>
        <textarea 
          className="w-full h-64 p-2 border rounded shadow-sm font-mono text-sm"
          value={emailInput}
          onChange={(e) => setEmailInput(e.target.value)}
        />
      </div>
      
      <button 
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 mb-4"
        onClick={runTest}
        disabled={loading}
      >
        {loading ? "Running..." : "Test Email Assistant"}
      </button>
      
      <div className="mb-4">
        <h2 className="text-xl font-semibold mb-2">Classification Result</h2>
        <div className="p-4 bg-gray-100 rounded">
          {output.classification_decision && (
            <div className="text-lg font-medium">
              {output.classification_decision === "respond" && "📧 RESPOND - This email requires a response"}
              {output.classification_decision === "ignore" && "🚫 IGNORE - This email can be safely ignored"}
              {output.classification_decision === "notify" && "🔔 NOTIFY - This email contains important information"}
            </div>
          )}
        </div>
      </div>
      
      {messages.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold mb-2">Messages</h2>
          <div className="space-y-4">
            {messages.map((message, index) => (
              <div key={index} className="p-4 border rounded">
                <div className="font-medium">{message.constructor.name}</div>
                <div className="whitespace-pre-wrap">{String(message.content)}</div>
                {message.additional_kwargs?.tool_calls && (
                  <div className="mt-2">
                    <div className="font-medium">Tool Calls:</div>
                    <pre className="bg-gray-100 p-2 rounded overflow-auto text-sm">
                      {JSON.stringify(message.additional_kwargs.tool_calls, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      
      <div className="mt-4">
        <h2 className="text-xl font-semibold mb-2">Full Output</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-auto text-sm">
          {JSON.stringify(output, null, 2)}
        </pre>
      </div>
    </div>
  );
} 