import { NextRequest, NextResponse } from "next/server";
import { getEmailAssistant } from "../../page";
import { EmailData } from "../../../lib/schemas";

// This is a temporary API route for testing the email assistant (before setting up LangSmith)
export async function POST(request: NextRequest) {
  try {
    const data = await request.json();
    const { email_input } = data;
    
    // Get the email assistant
    const emailAssistant = await getEmailAssistant();
    
    // Run the email assistant with the raw email text
    const result = await emailAssistant.invoke({
      email_input
    });
    
    return NextResponse.json(result);
  } catch (error: any) {
    console.error("Error running email assistant:", error);
    return NextResponse.json(
      { error: error.message }, 
      { status: 500 }
    );
  }
}