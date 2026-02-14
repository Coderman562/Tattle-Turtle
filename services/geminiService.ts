import { GoogleGenAI, Type } from "@google/genai";
import { EscalationType, PatternTracker, TurtleResponse, UrgencyLevel } from "../types";
import { SCHOOL_GUIDELINES } from "../constants";
import {
  generateWarmClosing,
  shouldContinueConversation,
  shouldEndConversation,
} from "../utils/conversationSignals";
import {
  determineEscalation,
  detectImmediateDangerOverride,
  getCurrentConcernType,
} from "../utils/escalationLogic";
import {
  getStudentPatterns,
  hasRecentEscalation,
  logInternalConcern,
  updatePatternHistory,
} from "../utils/patternTracking";

const DEFAULT_STUDENT_ID = "anonymous-student";

const placeholderExercise = {
  task: "Take one deep breath and wiggle your shoulders.",
  reward: "Brave Breathing Badge",
};

export const getTurtleSupport = async (
  userInput: string,
  conversationHistory?: string,
  studentPatternHistory?: PatternTracker[],
  studentId: string = DEFAULT_STUDENT_ID,
): Promise<TurtleResponse> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const fullTranscript = conversationHistory || userInput;

  const shouldExit = shouldEndConversation(userInput, fullTranscript);
  if (shouldExit) {
    return {
      sufficient: true,
      shouldEndConversation: true,
      closingMessage: generateWarmClosing(userInput),
      listeningHealing: "Thanks for sharing with me.",
      reflectionHelper: "",
      exercise: placeholderExercise,
      summary: "Conversation closed with a positive signal.",
      urgency: UrgencyLevel.GREEN,
      escalationType: EscalationType.NONE,
      needsEscalationConfirmation: false,
      concernType: getCurrentConcernType(fullTranscript),
    };
  }

  const shouldContinue = shouldContinueConversation(userInput, fullTranscript, false);
  const patterns = studentPatternHistory || getStudentPatterns(studentId);

  let escalation = determineEscalation(userInput, fullTranscript, patterns);
  const immediateDanger = detectImmediateDangerOverride(fullTranscript);
  const cooldownActive = hasRecentEscalation(studentId, 24);

  if (escalation === EscalationType.IMMEDIATE && cooldownActive && !immediateDanger) {
    escalation = EscalationType.PATTERN;
    logInternalConcern(studentId, "Immediate escalation suppressed due to cooldown.", EscalationType.PATTERN);
  }

  const concernType = getCurrentConcernType(fullTranscript);
  const urgency =
    escalation === EscalationType.IMMEDIATE
      ? UrgencyLevel.RED
      : escalation === EscalationType.PATTERN
        ? UrgencyLevel.YELLOW
        : UrgencyLevel.GREEN;

  const needsEscalationConfirmation = escalation === EscalationType.IMMEDIATE && !immediateDanger;

  const prompt = `
    You are Tattle Turtle, a gentle and supportive companion for children aged 6-10.

    CONVERSATION LOG:
    "${fullTranscript}"

    DETECTED SIGNALS:
    - Should continue gathering information: ${shouldContinue}
    - Escalation type: ${escalation}
    - Urgency level: ${urgency}
    - Concern category: ${concernType}

    RESPONSE RULES:
    - Keep language child-safe and age-appropriate.
    - Keep listeningHealing to 1 short sentence.
    - If should continue gathering information is true, sufficient must be false and followUpQuestion must ask only one gentle missing piece.
    - If urgency is RED and confirmation is needed, set helpInstruction to: "This sounds really important. I think your teacher should know so they can help. Is that okay with you?"
    - If urgency is RED and immediate danger is true, use helpInstruction to ask student to find a trusted adult right now.
    - Never include full transcript in summary.
    - Keep summary under 20 words.

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
                reward: { type: Type.STRING },
              },
              required: ["task", "reward"],
            },
            summary: { type: Type.STRING },
            urgency: {
              type: Type.STRING,
              description: "Must be GREEN, YELLOW, or RED",
            },
            helpInstruction: { type: Type.STRING },
            followUpNeeded: { type: Type.BOOLEAN },
          },
          required: ["sufficient", "listeningHealing", "reflectionHelper", "exercise", "summary", "urgency"],
        },
      },
    });

    const text = response.text || "{}";
    const parsed = JSON.parse(text) as TurtleResponse;

    const normalized: TurtleResponse = {
      ...parsed,
      sufficient: shouldContinue ? false : parsed.sufficient,
      followUpQuestion: shouldContinue
        ? parsed.followUpQuestion || "Can you tell me a little more about what happened?"
        : parsed.followUpQuestion,
      urgency,
      escalationType: escalation,
      needsEscalationConfirmation,
      shouldEndConversation: false,
      concernType,
      exercise: parsed.exercise || placeholderExercise,
      summary: (parsed.summary || "Student shared a concern and received support.").split(/\s+/).slice(0, 20).join(' '),
    };

    updatePatternHistory(studentId, concernType, normalized.summary);

    if (normalized.escalationType === EscalationType.IMMEDIATE && immediateDanger) {
      normalized.needsEscalationConfirmation = false;
      normalized.helpInstruction =
        normalized.helpInstruction || "Please find a trusted teacher or adult right now so they can keep you safe.";
    }

    return normalized;
  } catch (error) {
    console.error("Gemini Error:", error);
    throw new Error("Tattle Turtle is having a little bubble bath. Try again soon!");
  }
};
