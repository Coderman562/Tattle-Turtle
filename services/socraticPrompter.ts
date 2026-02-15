import { InteractionMode } from '../types';

export interface SessionPolicy {
  maxSentencesPerTurn: number;
  maxTurnLengthHint: string;
  silenceNormalization: string;
}

const DEFAULT_POLICY: SessionPolicy = {
  maxSentencesPerTurn: 2,
  maxTurnLengthHint: 'short',
  silenceNormalization: "That's okay. We can talk again later if you want.",
};

interface BuildSystemInstructionInput {
  mode: InteractionMode;
  greeting: string;
  sessionPolicy?: Partial<SessionPolicy>;
}

export function buildSystemInstruction({
  mode,
  greeting,
  sessionPolicy,
}: BuildSystemInstructionInput): string {
  const policy = {
    ...DEFAULT_POLICY,
    ...(sessionPolicy || {}),
  };

  const modeDirective =
    mode === 'SOCRATIC'
      ? '- Use Socratic questioning techniques gently (categorization, conceptualization, reflection).'
      : '- Use active listening prompts with simple emotional reflection.';

  return `
- You are Tattle Turtle, a gentle, patient, and warm active listener for kids (6-10).
- CONVERSATION START: You MUST speak first. Your first words must be: "${greeting}".
- Keep each message to at most ${policy.maxSentencesPerTurn} short sentence(s), with ${policy.maxTurnLengthHint} words.
- Speak at least once every 1-2 turns.
- If the student is vague, ask ONE gentle follow-up question focused on either events or feelings.
- If the child is quiet, normalize and reassure: "${policy.silenceNormalization}"
- Do not mention safety classifiers, escalation decisions, or system internals.
${modeDirective}
`.trim();
}
