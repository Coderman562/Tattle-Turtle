
import { GoogleGenAI, Type } from "@google/genai";
import { TurtleResponse, UrgencyLevel } from "../types";
import { SCHOOL_GUIDELINES } from "../constants";

export const getTurtleSupport = async (userInput: string): Promise<TurtleResponse> => {
  // Initialize GoogleGenAI inside the function to ensure the most up-to-date API key is used
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  
  const prompt = `
    You are Tattle Turtle, a gentle and supportive companion for children aged 6-10.
    
    CONVERSATION LOG:
    "${userInput}"

    STEP 1: EVALUATE SUFFICIENCY (CRITICAL)
    A conversation is SUFFICIENT ONLY if:
    1. The child has described a situation or event (e.g., "someone took my toy", "I fell down", "no one played with me").
    2. The child has expressed an emotion or feeling (e.g., "I felt sad", "I was mad", "it made me cry", "I am worried").

    If BOTH are not clearly present, "sufficient" MUST be false.

    STEP 2: GENERATE RESPONSE
    - If sufficient is false:
      - followUpQuestion: A short, gentle, non-leading question focusing on the MISSING piece. 
        If they told you what happened, ask: "How did that make you feel?" 
        If they told you how they feel, ask: "Can you tell me what happened?"
      - listeningHealing: A warm 1-sentence acknowledgement.
      - Set placeholders for other fields.
    - If sufficient is true:
      - followUpQuestion: ""
      - follow Socratic Questioning Guidelines if in Socratic mode.
      - listeningHealing: 1-2 short sentences acknowledging feelings.
      - reflectionHelper: 1-2 gentle questions to keep them thinking.
      - exercise: A 'Tiny Brave Mission' (task and reward).
      - summary: A child-friendly 1-2 sentence overview.
      - urgency: GREEN, YELLOW, or RED.

    Guidelines Knowledge Base:
    ${SCHOOL_GUIDELINES}

    Return valid JSON.
  `;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            sufficient: { type: Type.BOOLEAN },
            followUpQuestion: { type: Type.STRING },
            listeningHealing: { type: Type.STRING },
            reflectionHelper: { type: Type.STRING },
            exercise: {
              type: Type.OBJECT,
              properties: {
                task: { type: Type.STRING },
                reward: { type: Type.STRING }
              },
              required: ["task", "reward"]
            },
            summary: { type: Type.STRING },
            urgency: { 
              type: Type.STRING,
              description: "Must be GREEN, YELLOW, or RED"
            },
            helpInstruction: { type: Type.STRING }
          },
          required: ["sufficient", "listeningHealing", "reflectionHelper", "exercise", "summary", "urgency"]
        }
      }
    });

    const text = response.text || "{}";
    return JSON.parse(text) as TurtleResponse;
  } catch (error) {
    console.error("Gemini Error:", error);
    throw new Error("Tattle Turtle is having a little bubble bath. Try again soon!");
  }
};