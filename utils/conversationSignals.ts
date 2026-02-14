export function detectEmotionalResolution(transcript: string): boolean {
  const resolutionPhrases = [
    /\b(feel|feeling)\s+(better|good|okay|fine)\b/i,
    /\b(it'?s|its)\s+(okay|fine|alright|good)\s+now\b/i,
    /\bknow\s+what\s+to\s+do\b/i,
    /\bthat\s+(helps|helped)\b/i,
    /\bnot\s+(worried|sad|scared|mad)\s+anymore\b/i,
  ];

  return resolutionPhrases.some((pattern) => pattern.test(transcript));
}

export function detectActionCommitment(transcript: string): boolean {
  const commitmentPhrases = [
    /\bI'?ll\s+try\b/i,
    /\bI\s+can\s+do\s+that\b/i,
    /\bI\s+will\b/i,
    /\btomorrow\s+I'?ll\b/i,
    /\bI'?m\s+going\s+to\b/i,
    /\bokay,?\s+I'?ll\b/i,
  ];

  return commitmentPhrases.some((pattern) => pattern.test(transcript));
}

export function detectLowUrgencyIsolated(transcript: string): boolean {
  const repetitionIndicators = /\b(always|every|again|keeps|never stops|all the time)\b/i;
  const highIntensityWords = /\b(terrified|hate|can'?t stand|worst|horrible)\b/i;

  const hasRepetition = repetitionIndicators.test(transcript);
  const hasHighIntensity = highIntensityWords.test(transcript);

  return !hasRepetition && !hasHighIntensity;
}

export function detectExplicitStop(transcript: string): boolean {
  const stopPhrases = [
    /\bdon'?t\s+want\s+to\s+talk\b/i,
    /\bthat'?s\s+all\b/i,
    /\bI'?m\s+done\b/i,
    /\bcan\s+I\s+go\b/i,
    /\bnever\s?mind\b/i,
    /\bforget\s+it\b/i,
  ];

  return stopPhrases.some((pattern) => pattern.test(transcript));
}

export function shouldEndConversation(transcript: string, conversationHistory: string): boolean {
  return (
    detectEmotionalResolution(transcript) ||
    detectActionCommitment(transcript) ||
    (detectLowUrgencyIsolated(conversationHistory) && detectEmotionalResolution(transcript)) ||
    detectExplicitStop(transcript)
  );
}

export function hasCompleteSituationAndEmotion(transcript: string): boolean {
  const hasSituation = /\b(happened|did|took|said|went|was|were)\b/i.test(transcript);
  const emotionWords =
    /\b(sad|mad|angry|scared|worried|happy|frustrated|upset|hurt|lonely|left out|embarrassed)\b/i;
  const hasEmotion = emotionWords.test(transcript);

  return hasSituation && hasEmotion;
}

export function shouldContinueForInformation(conversationHistory: string): boolean {
  return !hasCompleteSituationAndEmotion(conversationHistory);
}

export function detectAmbiguousLanguage(transcript: string): boolean {
  const ambiguousPhrases = [
    /\bit\s+was\s+(weird|strange|bad|okay|fine)\b/i,
    /\bsomething\s+happened\b/i,
    /\bI\s+don'?t\s+know\b/i,
    /\bjust\.\.\.|um+|uh+/i,
    /\bkinda|sorta|like\b/i,
  ];

  return ambiguousPhrases.some((pattern) => pattern.test(transcript));
}

export function detectEmotionalInconsistency(transcript: string): boolean {
  const minimizingWords = /\b(fine|okay|whatever|no big deal)\b/i;
  const distressingContent = /\b(hurt|upset|crying|yelled|hit|pushed|left out|alone)\b/i;

  return minimizingWords.test(transcript) && distressingContent.test(transcript);
}

export function shouldContinueConversation(
  transcript: string,
  conversationHistory: string,
  silenceDetected: boolean,
): boolean {
  return (
    shouldContinueForInformation(conversationHistory) ||
    detectAmbiguousLanguage(transcript) ||
    detectEmotionalInconsistency(transcript) ||
    silenceDetected
  );
}

export function generateWarmClosing(transcript: string): string {
  if (detectActionCommitment(transcript)) {
    return "You have a brave plan. See you soon!";
  }

  if (detectExplicitStop(transcript)) {
    return "Thanks for sharing. We can talk again later.";
  }

  return "I'm glad we talked. Take care!";
}
