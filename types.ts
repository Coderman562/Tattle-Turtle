
export enum UrgencyLevel {
  GREEN = 'GREEN',
  YELLOW = 'YELLOW',
  RED = 'RED'
}

export interface ExerciseData {
  task: string;
  reward: string;
}

export interface TurtleResponse {
  sufficient: boolean;
  followUpQuestion?: string;
  listeningHealing: string;
  reflectionHelper: string;
  exercise: ExerciseData;
  summary: string;
  urgency: UrgencyLevel;
  helpInstruction?: string;
}

export type InteractionMode = 'LISTENING' | 'SOCRATIC';

export interface SessionState {
  step: 'WELCOME' | 'INPUT' | 'VOICE_CHAT' | 'PROCESSING' | 'RESULTS';
  inputMode: 'TEXT' | 'VOICE';
  interactionMode: InteractionMode;
  transcript: string;
  response: TurtleResponse | null;
}
