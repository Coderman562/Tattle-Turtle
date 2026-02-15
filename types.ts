export enum UrgencyLevel {
  GREEN = 'GREEN',
  YELLOW = 'YELLOW',
  RED = 'RED',
}

export enum EscalationType {
  NONE = 'NONE',
  PATTERN = 'PATTERN',
  IMMEDIATE = 'IMMEDIATE',
}

export type SafetyOutcome = 'GREEN' | 'TEACHER_REQUIRED';

export type ConcernType =
  | 'peer_conflict'
  | 'social_exclusion'
  | 'academic_stress'
  | 'family_conflict'
  | 'physical_complaint'
  | 'emotional_regulation';

export interface PatternTracker {
  studentId: string;
  concernType: ConcernType;
  occurrences: Array<{ date: string; summary: string }>;
  count: number;
}

export interface ExerciseData {
  task: string;
  reward: string;
}

export interface ConversationSummary {
  sufficient: boolean;
  shouldEndConversation?: boolean;
  closingMessage?: string;
  followUpQuestion?: string;
  listeningHealing: string;
  reflectionHelper: string;
  exercise: ExerciseData;
  summary: string;
  tammyResponse?: string;
  tags?: string[];
  teacherNote?: string;
  nextAction?: 'LISTEN' | 'FOLLOW_UP' | 'ESCALATE';
}

// Legacy response shape preserved for compatibility while conversation flow migrates.
export interface TurtleResponse extends ConversationSummary {
  needsEscalationConfirmation?: boolean;
  escalationType?: EscalationType;
  concernType?: ConcernType;
  urgency: UrgencyLevel;
  helpInstruction?: string;
  severity?: UrgencyLevel;
  emotionSource?: {
    verbal: string;
    facial: string;
    confidence: number;
    mismatch: boolean;
  };
  nextAction?: 'LISTEN' | 'FOLLOW_UP' | 'ESCALATE';
}

export type InteractionMode = 'LISTENING' | 'SOCRATIC';

export interface TeacherAlert {
  timestamp: string;
  safetyOutcome: SafetyOutcome;
  summary: string;
  reasonCode: string;
  teacherNotice?: string;
  actionSuggestion?: string;
}

export interface SessionState {
  step: 'WELCOME' | 'VOICE_CHAT' | 'PROCESSING' | 'RESULTS';
  interactionMode: InteractionMode;
  response: ConversationSummary | null;
}

export interface TurtleConversation {
  timestamp: string;
  studentText: string;
  turtleSummary: string;
  urgency: UrgencyLevel;
  concernType: ConcernType;
  escalationType: EscalationType;
  tags?: string[];
}

export interface StudentInfo {
  id: string;
  firstName?: string;
  parentEmail?: string;
  optedOutOfParentCommunication?: boolean;
  doNotContactParents?: boolean;
}

export interface ReadingMaterial {
  title: string;
  intro: string;
  quickRead: string;
  tips: string[];
  parentScript: string;
}

export interface Activity {
  title: string;
  durationMinutes: number;
  materials: string[];
  steps: string[];
  connectionQuestion: string;
}

export interface BookRecommendation {
  title: string;
  author: string;
  theme: string;
  ratingOutOf5: number;
  whyItFits: string;
}

export interface GrowthMoment {
  headline: string;
  celebration: string;
  skillsPracticed: string[];
  brightSpots: string[];
  encouragement: string;
}

export interface ParentSummary {
  readingMaterial: ReadingMaterial;
  activity: Activity;
  activities: Activity[];
  bookRecommendations: BookRecommendation[];
  growthMoment: GrowthMoment;
  weekCovered: string;
  generatedAt: string;
}

export type ConversationEventType =
  | 'SESSION_START'
  | 'STUDENT_UTTERANCE'
  | 'MODEL_UTTERANCE'
  | 'SESSION_END'
  | 'MANUAL_TEACHER_REQUEST';

export type ConversationRole = 'SYSTEM' | 'STUDENT' | 'MODEL';

export interface ConversationEvent {
  type: ConversationEventType;
  role: ConversationRole;
  studentId: string;
  conversationId: string;
  utteranceId: number;
  timestamp: string;
  text?: string;
  metadata?: Record<string, string | number | boolean>;
}

export interface SafetyDecision {
  studentId: string;
  conversationId: string;
  utteranceId: number;
  safetyOutcome: SafetyOutcome;
  teacherNotifyNow: boolean;
  shouldEndConversation: boolean;
  reasonCode: string;
  confidence: number;
  latencyMs?: number;
  studentNotice?: string;
  teacherNotice?: string;
}

export interface EvaluatorPolicyConfig {
  rollingWindowLimit: number;
  responseTimeoutMs: number;
  model: string;
  failSafeOutcome: SafetyOutcome;
}

export type EvaluatorWorkerMessageKind = 'EVALUATE' | 'DECISION' | 'WORKER_ERROR' | 'HEARTBEAT';

export interface EvaluatorWorkerMessage<T = unknown> {
  kind: EvaluatorWorkerMessageKind;
  payload: T;
}
