import { ConversationEvent } from '../types';

function newConversationId(studentId: string): string {
  return `${studentId}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export class ConversationEventEmitter {
  private conversationId = '';
  private utteranceId = 0;

  getConversationId(): string {
    return this.conversationId;
  }

  startSession(studentId: string): ConversationEvent {
    this.conversationId = newConversationId(studentId);
    this.utteranceId = 0;
    return {
      type: 'SESSION_START',
      role: 'SYSTEM',
      studentId,
      conversationId: this.conversationId,
      utteranceId: this.utteranceId,
      timestamp: new Date().toISOString(),
    };
  }

  studentUtterance(studentId: string, text: string): ConversationEvent {
    this.utteranceId += 1;
    return {
      type: 'STUDENT_UTTERANCE',
      role: 'STUDENT',
      studentId,
      conversationId: this.conversationId,
      utteranceId: this.utteranceId,
      timestamp: new Date().toISOString(),
      text,
    };
  }

  modelUtterance(studentId: string, text: string): ConversationEvent {
    this.utteranceId += 1;
    return {
      type: 'MODEL_UTTERANCE',
      role: 'MODEL',
      studentId,
      conversationId: this.conversationId,
      utteranceId: this.utteranceId,
      timestamp: new Date().toISOString(),
      text,
    };
  }

  manualTeacherRequest(studentId: string, text: string): ConversationEvent {
    this.utteranceId += 1;
    return {
      type: 'MANUAL_TEACHER_REQUEST',
      role: 'STUDENT',
      studentId,
      conversationId: this.conversationId,
      utteranceId: this.utteranceId,
      timestamp: new Date().toISOString(),
      text,
    };
  }

  endSession(studentId: string): ConversationEvent {
    this.utteranceId += 1;
    return {
      type: 'SESSION_END',
      role: 'SYSTEM',
      studentId,
      conversationId: this.conversationId,
      utteranceId: this.utteranceId,
      timestamp: new Date().toISOString(),
    };
  }
}
