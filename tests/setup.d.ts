import { ChatOpenAI } from "@langchain/openai";

declare global {
  var criteriaEvalLLM: ChatOpenAI;
}

export {};
