import { Type } from '@google/genai';
import { EvaluatorPolicyConfig } from '../types';

export const EVALUATOR_POLICY: EvaluatorPolicyConfig = {
  rollingWindowLimit: 4000,
  responseTimeoutMs: 12000,
  model: 'gemini-2.5-flash',
  failSafeOutcome: 'GREEN',
};

export const EVALUATOR_RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    safetyOutcome: {
      type: Type.STRING,
      description: 'Must be GREEN or TEACHER_REQUIRED',
    },
    shouldEndConversation: {
      type: Type.BOOLEAN,
    },
    reasonCode: {
      type: Type.STRING,
      description: 'Short snake_case reason code',
    },
    confidence: {
      type: Type.NUMBER,
      description: '0.0-1.0',
    },
    evidenceQuote: {
      type: Type.STRING,
      description: 'Direct quote from student content supporting the decision',
    },
    studentNotice: {
      type: Type.STRING,
    },
    teacherNotice: {
      type: Type.STRING,
    },
  },
  required: ['safetyOutcome', 'shouldEndConversation', 'reasonCode', 'confidence'],
};

interface BuildEvaluatorPromptInput {
  transcript: string;
  latestStudentUtterance: string;
}

export function buildEvaluatorPrompt({
  transcript,
  latestStudentUtterance,
}: BuildEvaluatorPromptInput): string {
  return `
You are a school safety evaluator for a child support conversation.

Task:
- Evaluate the FULL conversation context and latest student utterance.
- Return only JSON.
- Choose exactly one safety outcome:
  - GREEN: no immediate teacher intervention required.
  - TEACHER_REQUIRED: student must talk with a teacher now.

Rules:
- Base the decision on explicit text/content only (stated intent, plan, actions, and targets).
- Do NOT use emotional tone, style, politeness, or intensity as the escalation trigger.
- Choose TEACHER_REQUIRED only when the student shows:
  1) explicit self-harm ideation, intent, or plan, or
  2) credible intent or plan to commit a serious illegal act causing harm (for example violence, weapon use, or other clear law-breaking harm).
- Otherwise choose GREEN, including sadness, conflict, frustration, vague fear, or non-urgent help-seeking.
- TEACHER_REQUIRED examples:
  - "I am going to kill myself after school."
  - "I don't want to be alive anymore."
  - "I keep thinking about killing myself."
  - "I want to hurt myself."
  - "I brought a knife and I will stab him today."
  - "I'm going to bring a gun to school."
  - "We are going to beat him up at recess."
  - "I plan to jump off the roof."
- GREEN examples:
  - "I feel sad and mad."
  - "Someone was mean to me at lunch."
  - "I hate school and I am frustrated."
  - "I need help because I feel overwhelmed."
  - "I am scared about my test."
  - "Sometimes I get angry but I don't want to hurt anyone."
- If content is ambiguous and does not include clear self-harm intent/plan or serious illegal harm intent/plan, choose GREEN.
- reasonCode rules:
  - If safetyOutcome is TEACHER_REQUIRED, reasonCode MUST be exactly one of:
    - self_harm_intent_or_plan
    - serious_illegal_harm_intent_or_plan
  - If safetyOutcome is GREEN, use a non-escalation reasonCode (for example: no_clear_harm_intent, emotional_distress_only, ambiguous_content).
- evidenceQuote rules:
  - If safetyOutcome is TEACHER_REQUIRED, include a short direct quote from student text proving the threshold.
  - If safetyOutcome is GREEN, evidenceQuote can be empty.
- shouldEndConversation is independent from safety outcome:
  - true if student is done, resolved, or wants to stop.
  - false if conversation should continue.
- If safetyOutcome is TEACHER_REQUIRED, teacherNotice should be actionable and calm.
- studentNotice should be child-safe and brief when present.
- reasonCode must be a stable snake_case identifier.
- confidence must be between 0 and 1.

Latest student utterance:
"""${latestStudentUtterance}"""

Full rolling conversation transcript:
"""${transcript}"""
  `.trim();
}
