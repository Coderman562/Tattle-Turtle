import {
  EscalationType,
  InteractionMode,
  PatternTracker,
  TurtleResponse,
  UrgencyLevel,
} from '../types';
import { getConversationSummary } from './conversationSummaryService';

const DEFAULT_STUDENT_ID = 'anonymous-student';

// Legacy compatibility wrapper. Conversation safety decisions are no longer produced here.
export const getTurtleSupport = async (
  userInput: string,
  conversationHistory?: string,
  _studentPatternHistory?: PatternTracker[],
  _studentId: string = DEFAULT_STUDENT_ID,
  _facialEmotionInput?: unknown,
  interactionMode: InteractionMode = 'LISTENING',
): Promise<TurtleResponse> => {
  const summary = await getConversationSummary(
    userInput,
    conversationHistory || userInput,
    interactionMode,
  );

  return {
    ...summary,
    urgency: UrgencyLevel.GREEN,
    escalationType: EscalationType.NONE,
    concernType: 'emotional_regulation',
    needsEscalationConfirmation: false,
  };
};
