
export enum UrgencyLevel {
  GREEN = 'GREEN',
  YELLOW = 'YELLOW',
  RED = 'RED'
}

export enum EscalationType {
  NONE = 'NONE',
  PATTERN = 'PATTERN',
  IMMEDIATE = 'IMMEDIATE'
}

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

export interface TurtleResponse {
  sufficient: boolean;
  shouldEndConversation?: boolean;
  closingMessage?: string;
  needsEscalationConfirmation?: boolean;
  escalationType?: EscalationType;
  concernType?: ConcernType;
  followUpQuestion?: string;
  listeningHealing: string;
  reflectionHelper: string;
  exercise: ExerciseData;
  summary: string;
  urgency: UrgencyLevel;
  helpInstruction?: string;
  tammyResponse?: string;
  severity?: UrgencyLevel;
  tags?: string[];
  emotionSource?: {
    verbal: string;
    facial: string;
    confidence: number;
    mismatch: boolean;
  };
  teacherNote?: string;
  nextAction?: 'LISTEN' | 'FOLLOW_UP' | 'ESCALATE';
}

export type InteractionMode = 'LISTENING' | 'SOCRATIC';

export interface TeacherAlert {
  timestamp: string;
  escalationType: EscalationType;
  urgency: UrgencyLevel;
  summary: string;
  concernCategory: ConcernType;
  primaryEmotion: string;
  patternFlag: boolean;
  studentConfirmedEscalation: boolean;
  actionSuggestion?: string;
}

export interface SessionState {
  step: 'WELCOME' | 'INPUT' | 'VOICE_CHAT' | 'PROCESSING' | 'RESULTS';
  inputMode: 'TEXT' | 'VOICE';
  interactionMode: InteractionMode;
  transcript: string;
  response: TurtleResponse | null;
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
