/// <reference lib="webworker" />

import { GoogleGenAI } from '@google/genai';
import {
  ConversationEvent,
  EvaluatorWorkerMessage,
  SafetyDecision,
  SafetyOutcome,
} from '../types';
import {
  buildEvaluatorPrompt,
  EVALUATOR_POLICY,
  EVALUATOR_RESPONSE_SCHEMA,
} from '../services/evaluatorPolicy';
import { detectCriticalTeacherRequiredEvidence } from '../services/criticalSafetyDetector';

type TranscriptTurn = {
  role: 'Student' | 'Turtle' | 'System';
  text: string;
};

type WorkerState = {
  studentId: string;
  transcript: TranscriptTurn[];
};

const stateByConversation = new Map<string, WorkerState>();

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

function compactTranscript(turns: TranscriptTurn[], limit: number): TranscriptTurn[] {
  if (limit <= 0) {
    return turns;
  }

  let total = turns.reduce((sum, turn) => sum + turn.text.length, 0);
  if (total <= limit) {
    return turns;
  }

  const trimmed = [...turns];
  while (trimmed.length > 1 && total > limit) {
    const removed = trimmed.shift();
    total -= removed?.text.length || 0;
  }

  if (trimmed.length === 1 && trimmed[0].text.length > limit) {
    trimmed[0] = {
      ...trimmed[0],
      text: trimmed[0].text.slice(-limit),
    };
  }

  return trimmed;
}

function toSafetyOutcome(value: string): SafetyOutcome {
  return value === 'TEACHER_REQUIRED' ? 'TEACHER_REQUIRED' : 'GREEN';
}

function normalizeConfidence(value: unknown): number {
  const num = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(num)) {
    return 0.5;
  }
  return Math.max(0, Math.min(1, num));
}

function normalizeForEvidenceMatch(value: string): string {
  return value
    .toLowerCase()
    .replace(/[’`]/g, "'")
    .replace(/[^a-z0-9'\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function baseDecision(event: ConversationEvent, outcome: SafetyOutcome, reasonCode: string): SafetyDecision {
  return {
    studentId: event.studentId,
    conversationId: event.conversationId,
    utteranceId: event.utteranceId,
    safetyOutcome: outcome,
    teacherNotifyNow: outcome === 'TEACHER_REQUIRED',
    shouldEndConversation: false,
    reasonCode,
    confidence: 1,
    studentNotice:
      outcome === 'TEACHER_REQUIRED'
        ? 'Teacher has been notified to check in privately.'
        : undefined,
    teacherNotice:
      outcome === 'TEACHER_REQUIRED' ? 'Immediate teacher check-in required.' : undefined,
  };
}

function renderTranscript(turns: TranscriptTurn[]): string {
  return turns.map((turn) => `${turn.role}: ${turn.text}`).join('\n');
}

function mapEventRole(event: ConversationEvent): TranscriptTurn['role'] {
  if (event.role === 'STUDENT') {
    return 'Student';
  }
  if (event.role === 'MODEL') {
    return 'Turtle';
  }
  return 'System';
}

function modelUnavailableDecision(
  event: ConversationEvent,
  startedAt: number,
  reasonCode: string,
): SafetyDecision {
  return {
    studentId: event.studentId,
    conversationId: event.conversationId,
    utteranceId: event.utteranceId,
    safetyOutcome: 'GREEN',
    teacherNotifyNow: false,
    shouldEndConversation: false,
    reasonCode,
    confidence: 0.35,
    latencyMs: Date.now() - startedAt,
    studentNotice: undefined,
    teacherNotice: undefined,
  };
}

async function evaluateWithModel(event: ConversationEvent, state: WorkerState): Promise<SafetyDecision> {
  const startedAt = Date.now();
  const transcript = renderTranscript(state.transcript);
  const latestStudentUtterance = event.text || '';
  const prompt = buildEvaluatorPrompt({
    transcript,
    latestStudentUtterance,
  });

  let responseText = '';
  try {
    const response = await ai.models.generateContent({
      model: EVALUATOR_POLICY.model,
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: EVALUATOR_RESPONSE_SCHEMA,
      },
    });
    responseText = response.text || '{}';
  } catch (error: any) {
    console.warn('[EvaluatorWorkerModelFallback]', String(error?.message || error));
    return modelUnavailableDecision(event, startedAt, 'model_unavailable_deterministic_only');
  }

  let parsed: Partial<{
    safetyOutcome: string;
    shouldEndConversation: boolean;
    reasonCode: string;
    confidence: number;
    evidenceQuote: string;
    studentNotice: string;
    teacherNotice: string;
  }>;
  try {
    parsed = JSON.parse(responseText || '{}');
  } catch (error: any) {
    console.warn('[EvaluatorWorkerInvalidJSON]', String(error?.message || error));
    return modelUnavailableDecision(event, startedAt, 'model_invalid_response');
  }

  const teacherRequiredReasonCodes = new Set([
    'self_harm_intent_or_plan',
    'serious_illegal_harm_intent_or_plan',
  ]);

  let safetyOutcome = toSafetyOutcome(String(parsed.safetyOutcome || 'GREEN'));
  const shouldEndConversation = Boolean(parsed.shouldEndConversation);
  let reasonCode = String(parsed.reasonCode || 'model_decision');
  const evidenceQuote = String(parsed.evidenceQuote || '').trim();
  const sourceText = normalizeForEvidenceMatch(`${latestStudentUtterance}\n${transcript}`);
  const normalizedEvidenceQuote = normalizeForEvidenceMatch(evidenceQuote);
  const hasEvidenceQuote =
    normalizedEvidenceQuote.length >= 6 && sourceText.includes(normalizedEvidenceQuote);
  const confidence = normalizeConfidence(parsed.confidence);
  const criticalMatch = detectCriticalTeacherRequiredEvidence(`${latestStudentUtterance}\n${transcript}`);

  if (
    safetyOutcome === 'TEACHER_REQUIRED' &&
    (!teacherRequiredReasonCodes.has(reasonCode) || (!hasEvidenceQuote && !criticalMatch.matched))
  ) {
    safetyOutcome = 'GREEN';
    reasonCode = 'insufficient_teacher_required_evidence';
  }

  return {
    studentId: event.studentId,
    conversationId: event.conversationId,
    utteranceId: event.utteranceId,
    safetyOutcome,
    teacherNotifyNow: safetyOutcome === 'TEACHER_REQUIRED',
    shouldEndConversation,
    reasonCode,
    confidence,
    latencyMs: Date.now() - startedAt,
    studentNotice: parsed.studentNotice,
    teacherNotice: parsed.teacherNotice,
  };
}

async function evaluateEvent(event: ConversationEvent): Promise<SafetyDecision> {
  const existing = stateByConversation.get(event.conversationId) || {
    studentId: event.studentId,
    transcript: [] as TranscriptTurn[],
  };

  if (event.type === 'SESSION_START') {
    stateByConversation.set(event.conversationId, existing);
    return baseDecision(event, 'GREEN', 'session_start');
  }

  if (event.type === 'SESSION_END') {
    return {
      ...baseDecision(event, 'GREEN', 'session_end'),
      shouldEndConversation: true,
    };
  }

  if (event.type === 'MANUAL_TEACHER_REQUEST') {
    return baseDecision(event, 'TEACHER_REQUIRED', 'manual_teacher_request');
  }

  const text = event.text?.trim();
  if (text) {
    existing.transcript.push({
      role: mapEventRole(event),
      text,
    });
    existing.transcript = compactTranscript(existing.transcript, EVALUATOR_POLICY.rollingWindowLimit);
  }
  stateByConversation.set(event.conversationId, existing);

  if (event.type === 'MODEL_UTTERANCE') {
    return baseDecision(event, 'GREEN', 'model_context_update');
  }

  // Deterministic hard trigger: explicit self-harm or serious illegal harm intent/plan.
  const combinedStudentText = existing.transcript
    .filter((turn) => turn.role === 'Student')
    .map((turn) => turn.text)
    .join('\n');
  const criticalMatch = detectCriticalTeacherRequiredEvidence(
    `${event.text || ''}\n${combinedStudentText}`.trim(),
  );
  if (criticalMatch.matched && criticalMatch.reasonCode) {
    return {
      studentId: event.studentId,
      conversationId: event.conversationId,
      utteranceId: event.utteranceId,
      safetyOutcome: 'TEACHER_REQUIRED',
      teacherNotifyNow: true,
      shouldEndConversation: true,
      reasonCode: criticalMatch.reasonCode,
      confidence: 1,
      studentNotice: 'Teacher has been notified to check in privately.',
      teacherNotice: criticalMatch.evidenceQuote
        ? `Critical safety statement detected: "${criticalMatch.evidenceQuote}".`
        : 'Critical safety statement detected.',
    };
  }

  return evaluateWithModel(event, existing);
}

function postMessageSafe<T>(message: EvaluatorWorkerMessage<T>): void {
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(message);
}

setInterval(() => {
  postMessageSafe({
    kind: 'HEARTBEAT',
    payload: { timestamp: Date.now() },
  });
}, 4000);

self.onmessage = async (raw: MessageEvent<EvaluatorWorkerMessage<ConversationEvent>>) => {
  try {
    const message = raw.data;
    if (!message || message.kind !== 'EVALUATE') {
      return;
    }

    const decision = await evaluateEvent(message.payload);
    postMessageSafe({ kind: 'DECISION', payload: decision });
  } catch (error: any) {
    postMessageSafe({
      kind: 'WORKER_ERROR',
      payload: {
        code: 'EVALUATION_FAILURE',
        details: String(error?.message || error || 'Unknown worker error'),
      },
    });
  }
};
