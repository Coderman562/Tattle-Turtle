import {
  ConversationEvent,
  EvaluatorWorkerMessage,
  SafetyDecision,
} from '../types';
import { EVALUATOR_POLICY } from './evaluatorPolicy';
import { detectCriticalTeacherRequiredEvidence } from './criticalSafetyDetector';

type DecisionHandler = (decision: SafetyDecision) => void;
type FailureHandler = (reason: string) => void;

export class EvaluatorClient {
  private worker: Worker | null = null;
  private onDecision: DecisionHandler;
  private onFailure?: FailureHandler;
  private responseTimeout: number | null = null;
  private failed = false;
  private lastEvent: ConversationEvent | null = null;

  constructor(onDecision: DecisionHandler, onFailure?: FailureHandler) {
    this.onDecision = onDecision;
    this.onFailure = onFailure;
  }

  init(): boolean {
    this.failed = false;
    this.lastEvent = null;

    try {
      this.worker = new Worker(new URL('../workers/conversationEvaluator.worker.ts', import.meta.url), {
        type: 'module',
      });
    } catch (error: any) {
      this.handleFailure(`Evaluator worker failed to initialize: ${String(error?.message || error)}`);
      return false;
    }

    this.worker.onmessage = (event: MessageEvent<EvaluatorWorkerMessage<any>>) => {
      const message = event.data;
      if (!message) {
        return;
      }

      if (message.kind === 'HEARTBEAT') {
        return;
      }

      if (message.kind === 'WORKER_ERROR') {
        this.handleFailure(`Evaluator worker error: ${message.payload?.details || 'unknown error'}`);
        return;
      }

      if (message.kind === 'DECISION') {
        this.clearResponseTimeout();
        this.onDecision(message.payload as SafetyDecision);
      }
    };

    this.worker.onerror = (event: ErrorEvent) => {
      this.handleFailure(`Evaluator worker crashed: ${event.message || 'unknown worker crash'}`);
    };
    return true;
  }

  emit(event: ConversationEvent): void {
    if (!this.worker || this.failed) {
      return;
    }
    this.lastEvent = event;
    this.worker.postMessage({
      kind: 'EVALUATE',
      payload: event,
    } satisfies EvaluatorWorkerMessage<ConversationEvent>);
    this.armResponseTimeout();
  }

  terminate(): void {
    this.clearResponseTimeout();
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
  }

  private armResponseTimeout(): void {
    this.clearResponseTimeout();
    this.responseTimeout = window.setTimeout(() => {
      this.handleFailure('Evaluator worker timeout. Triggering fail-safe escalation.');
    }, EVALUATOR_POLICY.responseTimeoutMs);
  }

  private clearResponseTimeout(): void {
    if (this.responseTimeout !== null) {
      window.clearTimeout(this.responseTimeout);
      this.responseTimeout = null;
    }
  }

  private handleFailure(reason: string): void {
    if (this.failed) {
      return;
    }
    this.failed = true;
    this.clearResponseTimeout();
    this.terminate();

    const fallbackConversationId = this.lastEvent?.conversationId || `fallback-${Date.now()}`;
    const fallbackStudentId = this.lastEvent?.studentId || 'unknown-student';
    const fallbackUtteranceId = this.lastEvent?.utteranceId || 0;

    const criticalMatch = detectCriticalTeacherRequiredEvidence(this.lastEvent?.text || '');
    const failSafeOutcome =
      criticalMatch.matched && criticalMatch.reasonCode
        ? 'TEACHER_REQUIRED'
        : EVALUATOR_POLICY.failSafeOutcome;

    this.onDecision({
      studentId: fallbackStudentId,
      conversationId: fallbackConversationId,
      utteranceId: fallbackUtteranceId,
      safetyOutcome: failSafeOutcome,
      teacherNotifyNow: failSafeOutcome === 'TEACHER_REQUIRED',
      shouldEndConversation: failSafeOutcome === 'TEACHER_REQUIRED',
      reasonCode:
        (criticalMatch.matched && criticalMatch.reasonCode) || 'evaluator_failure',
      confidence: 1,
      studentNotice: failSafeOutcome === 'TEACHER_REQUIRED'
        ? 'Teacher has been notified to check in privately.'
        : 'Safety evaluator is temporarily unavailable. Turtle will keep listening.',
      teacherNotice:
        failSafeOutcome === 'TEACHER_REQUIRED'
          ? criticalMatch.evidenceQuote
            ? `Critical safety statement detected: "${criticalMatch.evidenceQuote}".`
            : reason
          : undefined,
    });

    this.onFailure?.(reason);
  }
}
