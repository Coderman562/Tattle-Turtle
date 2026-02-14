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

type CameraEmotionLabel = "angry" | "disgust" | "fear" | "happy" | "sad" | "surprise" | "neutral";

type CameraEmotionInput =
  | string
  | {
      emotion: string;
      confidence: number;
    };

const CAMERA_EMOTIONS: CameraEmotionLabel[] = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"];

const emotionStreakByStudent = new Map<string, { sadStreak: number }>();

const placeholderExercise = {
  task: "Take one deep breath and wiggle your shoulders.",
  reward: "Brave Breathing Badge",
};

function normalizeConfidence(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }

  if (value > 1) {
    return Math.max(0, Math.min(1, value / 100));
  }

  return Math.max(0, Math.min(1, value));
}

function parseCameraEmotion(input?: CameraEmotionInput): { emotion: CameraEmotionLabel; confidence: number } {
  if (!input) {
    return { emotion: "neutral", confidence: 0 };
  }

  if (typeof input === "object") {
    const emotion = String(input.emotion || "neutral").toLowerCase();
    const confidence = normalizeConfidence(input.confidence);
    return {
      emotion: CAMERA_EMOTIONS.includes(emotion as CameraEmotionLabel) ? (emotion as CameraEmotionLabel) : "neutral",
      confidence,
    };
  }

  const raw = String(input).trim().toLowerCase();
  const emotionMatch = raw.match(/(angry|disgust|fear|happy|sad|surprise|neutral)/);
  const confidenceMatch = raw.match(/([0-9]+(?:\.[0-9]+)?)\s*%?/);

  const parsedConfidence = confidenceMatch ? normalizeConfidence(Number(confidenceMatch[1])) : 0;

  return {
    emotion: emotionMatch ? (emotionMatch[1] as CameraEmotionLabel) : "neutral",
    confidence: parsedConfidence,
  };
}

function inferVerbalEmotion(text: string): string {
  const normalized = text.toLowerCase();

  if (/\b(happy|great|awesome|fun|yay|excited|good)\b/.test(normalized)) return "joy";
  if (/\b(sad|cry|down|upset|lonely)\b/.test(normalized)) return "sadness";
  if (/\b(scared|afraid|nervous|worried|fear)\b/.test(normalized)) return "fear";
  if (/\b(angry|mad|furious|annoyed)\b/.test(normalized)) return "anger";
  if (/\b(disgust|gross|ew)\b/.test(normalized)) return "disgust";
  if (/\b(wow|surprised|unexpected)\b/.test(normalized)) return "surprise";
  if (/\b(i'?m\s*fine|okay|ok|nothing)\b/.test(normalized)) return "neutral";

  return "neutral";
}

function hasHarmMention(text: string): boolean {
  return /\b(hit|hitting|hurt|hurting|weapon|gun|knife|bleeding|pushed|abuse)\b/i.test(text);
}

function mentionsSpecificPerson(text: string): boolean {
  return /\b(my\s+(friend|teacher|mom|dad|brother|sister)|he|she|they|someone)\b/i.test(text);
}

function computeCameraEscalation(
  studentId: string,
  voiceTranscript: string,
  cameraEmotion: CameraEmotionLabel,
  cameraConfidence: number,
): UrgencyLevel {
  const streak = emotionStreakByStudent.get(studentId) || { sadStreak: 0 };

  if (cameraEmotion === "sad" && cameraConfidence > 0.7) {
    streak.sadStreak += 1;
  } else {
    streak.sadStreak = 0;
  }

  emotionStreakByStudent.set(studentId, streak);

  if (cameraEmotion === "fear" && cameraConfidence > 0.8 && hasHarmMention(voiceTranscript)) {
    return UrgencyLevel.RED;
  }

  if (cameraEmotion === "fear" && cameraConfidence > 0.8) {
    return UrgencyLevel.YELLOW;
  }

  if (cameraEmotion === "sad" && cameraConfidence > 0.7 && streak.sadStreak >= 3) {
    return UrgencyLevel.YELLOW;
  }

  if (cameraEmotion === "angry" && cameraConfidence > 0.8 && mentionsSpecificPerson(voiceTranscript)) {
    return UrgencyLevel.YELLOW;
  }

  return UrgencyLevel.GREEN;
}

function maxUrgency(a: UrgencyLevel, b: UrgencyLevel): UrgencyLevel {
  const order = {
    [UrgencyLevel.GREEN]: 0,
    [UrgencyLevel.YELLOW]: 1,
    [UrgencyLevel.RED]: 2,
  } as const;

  return order[a] >= order[b] ? a : b;
}

export const getTurtleSupport = async (
  userInput: string,
  conversationHistory?: string,
  studentPatternHistory?: PatternTracker[],
  studentId: string = DEFAULT_STUDENT_ID,
  facialEmotionInput?: CameraEmotionInput,
): Promise<TurtleResponse> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const fullTranscript = conversationHistory || userInput;
  const { emotion: facialEmotion, confidence: facialConfidence } = parseCameraEmotion(facialEmotionInput);

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
      tags: ["positive-sharing"],
      emotionSource: {
        verbal: inferVerbalEmotion(userInput),
        facial: facialEmotion,
        confidence: facialConfidence,
        mismatch: false,
      },
      teacherNote: "",
      nextAction: "LISTEN",
      tammyResponse: "Thanks for sharing with me.",
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
  const baseUrgency =
    escalation === EscalationType.IMMEDIATE
      ? UrgencyLevel.RED
      : escalation === EscalationType.PATTERN
        ? UrgencyLevel.YELLOW
        : UrgencyLevel.GREEN;

  const cameraUrgency = computeCameraEscalation(studentId, fullTranscript, facialEmotion, facialConfidence);
  const urgency = maxUrgency(baseUrgency, cameraUrgency);

  if (urgency === UrgencyLevel.RED) {
    escalation = EscalationType.IMMEDIATE;
  } else if (urgency === UrgencyLevel.YELLOW && escalation === EscalationType.NONE) {
    escalation = EscalationType.PATTERN;
  }

  const needsEscalationConfirmation = escalation === EscalationType.IMMEDIATE && !immediateDanger;

  const prompt = `
You are Tammy the Tattle Turtle, a warm classroom companion for students ages 5-12.

You receive two inputs:
1) Voice transcript (student words)
2) Facial emotion from camera model trpakov/vit-face-expression in format emotion + confidence

INPUTS:
- Voice transcript: """${userInput}"""
- Facial emotion: "${facialEmotion}: ${Math.round(facialConfidence * 100)}%"

Follow these rules exactly:
- Respond warmly, age-appropriate, and short.
- tammyResponse must be max 15 words.
- Never reference camera, facial expression, or mismatch directly to the student.
- If confidence < 0.5 ignore camera for behavior and urgency.
- If confidence is 0.5 to 0.7, include camera context only in teacherNote.
- If confidence > 0.7, factor camera into mismatch + urgency.
- FEAR > 0.8 at any point means minimum YELLOW.
- FEAR > 0.8 plus harm/hitting/hurting/weapons mention means RED.
- SAD > 0.7 across 3+ consecutive readings means YELLOW.
- ANGER > 0.8 plus specific person mention means YELLOW.
- Camera can escalate urgency but never de-escalate text risk.

Mismatch rules:
- "I'm fine" with SAD/FEAR > 0.7: gentle invitation to share more.
- Minor event with ANGER > 0.7: ask if anything else happened.
- No harm mentioned but FEAR > 0.8: elevate urgency one level and gently check in.
- HAPPY with positive sharing: match positive energy.
- NEUTRAL: respond to words only.

Output must be valid JSON with this exact structure:
{
  "tammyResponse": "string max 15 words",
  "severity": "GREEN | YELLOW | RED",
  "tags": ["allowed tags only"],
  "emotionSource": {
    "verbal": "emotion from words",
    "facial": "raw camera emotion",
    "confidence": 0.0,
    "mismatch": true
  },
  "teacherNote": "one sentence only when mismatch=true or severity is YELLOW/RED, else empty",
  "nextAction": "LISTEN | FOLLOW_UP | ESCALATE"
}

Allowed tags:
friendship, family, school-stress, loneliness, anger, sadness, anxiety, bullying, self-esteem, grief, fear, physical-harm, abuse, self-harm, medical, conflict, positive-sharing, mismatch-detected

Conversation context:
"""${fullTranscript}"""

Detected platform signals:
- Concern category: ${concernType}
- Base urgency from transcript: ${baseUrgency}
- Final required urgency floor (after camera rule): ${urgency}
- Should continue gathering info: ${shouldContinue}

Guidelines Knowledge Base:
${SCHOOL_GUIDELINES}

Return JSON only.
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
            tammyResponse: { type: Type.STRING },
            severity: { type: Type.STRING, description: "Must be GREEN, YELLOW, or RED" },
            tags: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            emotionSource: {
              type: Type.OBJECT,
              properties: {
                verbal: { type: Type.STRING },
                facial: { type: Type.STRING },
                confidence: { type: Type.NUMBER },
                mismatch: { type: Type.BOOLEAN },
              },
              required: ["verbal", "facial", "confidence", "mismatch"],
            },
            teacherNote: { type: Type.STRING },
            nextAction: { type: Type.STRING },
          },
          required: ["tammyResponse", "severity", "tags", "emotionSource", "teacherNote", "nextAction"],
        },
      },
    });

    const text = response.text || "{}";
    const parsed = JSON.parse(text) as Partial<TurtleResponse> & {
      severity?: string;
      nextAction?: string;
    };

    const parsedUrgency =
      parsed.severity === UrgencyLevel.RED
        ? UrgencyLevel.RED
        : parsed.severity === UrgencyLevel.YELLOW
          ? UrgencyLevel.YELLOW
          : UrgencyLevel.GREEN;

    const finalUrgency = maxUrgency(parsedUrgency, urgency);

    const fallbackTammy = finalUrgency === UrgencyLevel.RED
      ? "Thanks for telling me. Let's get a trusted adult to help right now."
      : shouldContinue
        ? "I'm listening. Can you tell me a little more?"
        : "Thank you for sharing. I'm proud of you for telling me.";

    const tammyResponse = (parsed.tammyResponse || fallbackTammy)
      .split(/\s+/)
      .slice(0, 15)
      .join(" ");

    const nextAction = parsed.nextAction === "ESCALATE"
      ? "ESCALATE"
      : parsed.nextAction === "FOLLOW_UP"
        ? "FOLLOW_UP"
        : "LISTEN";

    const normalized: TurtleResponse = {
      ...parsed,
      tammyResponse,
      sufficient: shouldContinue ? false : nextAction !== "FOLLOW_UP",
      followUpQuestion:
        nextAction === "FOLLOW_UP"
          ? tammyResponse
          : shouldContinue
            ? "Can you tell me a little more about what happened?"
            : undefined,
      listeningHealing: tammyResponse,
      reflectionHelper: parsed.reflectionHelper || "Thanks for sharing with me.",
      urgency: finalUrgency,
      escalationType: finalUrgency === UrgencyLevel.RED ? EscalationType.IMMEDIATE : escalation,
      needsEscalationConfirmation,
      shouldEndConversation: false,
      concernType,
      exercise: parsed.exercise || placeholderExercise,
      summary: ((parsed.teacherNote as string) || tammyResponse || "Student shared a concern and received support.")
        .split(/\s+/)
        .slice(0, 20)
        .join(" "),
      teacherNote: (parsed.teacherNote as string) || "",
      nextAction,
      emotionSource: {
        verbal: parsed.emotionSource?.verbal || inferVerbalEmotion(userInput),
        facial: parsed.emotionSource?.facial || facialEmotion,
        confidence: normalizeConfidence(parsed.emotionSource?.confidence ?? facialConfidence),
        mismatch: Boolean(parsed.emotionSource?.mismatch),
      },
      tags: Array.isArray(parsed.tags) ? parsed.tags : [],
      helpInstruction:
        finalUrgency === UrgencyLevel.RED
          ? "This sounds really important. Please find a trusted teacher or adult right now."
          : parsed.helpInstruction,
    };

    if (
      normalized.emotionSource &&
      normalized.emotionSource.confidence > 0.7 &&
      normalized.emotionSource.mismatch &&
      !normalized.tags?.includes("mismatch-detected")
    ) {
      normalized.tags = [...(normalized.tags || []), "mismatch-detected"];
    }

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
