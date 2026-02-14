import { ConcernType, EscalationType, PatternTracker, TeacherAlert, TurtleResponse, UrgencyLevel } from '../types';

export function detectHighRiskContent(transcript: string): boolean {
  const highRiskPatterns = [
    /\b(hurt\s+myself|kill\s+myself|want\s+to\s+die|disappear|end\s+it)\b/i,
    /\b(hit\s+me|hurt\s+me|touched\s+me|bleeding|bruise)\b/i,
    /\b(gun|knife|weapon|threat|going\s+to\s+hurt)\b/i,
    /\b(always\s+hurt|keeps\s+hitting|won'?t\s+stop|every\s+day.*hurt)\b/i,
    /\b(scared\s+to\s+go\s+home|afraid\s+of.*school|don'?t\s+feel\s+safe)\b/i,
    /\b(can'?t\s+(take|handle)\s+it|nobody\s+cares|all\s+alone|no\s+one\s+helps)\b/i,
  ];

  return highRiskPatterns.some((pattern) => pattern.test(transcript));
}

export function detectRepeatedPattern(currentConcern: ConcernType, studentHistory: PatternTracker[]): boolean {
  const matchingPattern = studentHistory.find((pattern) => pattern.concernType === currentConcern);
  return Boolean(matchingPattern && matchingPattern.count >= 3);
}

export function detectHighEmotionalIntensity(transcript: string): boolean {
  const intensityPhrases = [
    /\bI'?m\s+(so\s+)?(scared|terrified|afraid)\b/i,
    /\bcan'?t\s+handle\s+it\b/i,
    /\bfeel\s+alone\s+all\s+the\s+time\b/i,
    /\balways\s+(sad|crying|upset)\b/i,
    /\bhate\s+(myself|it|everything)\b/i,
  ];

  return intensityPhrases.some((pattern) => pattern.test(transcript));
}

export function detectPowerImbalance(transcript: string): boolean {
  const powerImbalanceIndicators = [
    /\b(excluded|left\s+out|nobody\s+lets\s+me).*(always|every|again)\b/i,
    /\bolder\s+(kid|student).*(threaten|scare|force)\b/i,
    /\bmade\s+me\s+(do|say|give)\b/i,
    /\bwon'?t\s+let\s+me\s+(play|join|sit)\b.*\b(ever|never|always)\b/i,
  ];

  return powerImbalanceIndicators.some((pattern) => pattern.test(transcript));
}

export function detectHelpRequest(transcript: string): boolean {
  const helpPhrases = [
    /\bcan\s+you\s+tell\s+(the\s+)?teacher\b/i,
    /\bI\s+need\s+help\b/i,
    /\bdon'?t\s+know\s+what\s+to\s+do\b/i,
    /\bcan\s+someone\s+help\b/i,
    /\btell\s+(my\s+)?(teacher|mom|dad)\b/i,
  ];

  return helpPhrases.some((pattern) => pattern.test(transcript));
}

export function detectImmediateDangerOverride(transcript: string): boolean {
  const emergencyOnlyPatterns = [
    /\b(happening\s+right\s+now|right\s+now\s+he'?s\s+hitting|they'?re\s+hurting\s+me\s+now)\b/i,
    /\b(I\s+am\s+going\s+to\s+kill\s+myself|I'?m\s+going\s+to\s+kill\s+myself)\b/i,
    /\b(I\s+have\s+a\s+knife|someone\s+has\s+a\s+gun)\b/i,
  ];

  return emergencyOnlyPatterns.some((pattern) => pattern.test(transcript));
}

export function getCurrentConcernType(transcript: string): ConcernType {
  if (/\b(bully|mean|fight|pushed|hit|tease|exclude|left out|recess|friend)\b/i.test(transcript)) {
    return 'peer_conflict';
  }

  if (/\b(nobody\s+plays|no\s+one\s+picked\s+me|alone\s+at\s+lunch|left\s+out)\b/i.test(transcript)) {
    return 'social_exclusion';
  }

  if (/\b(test|grade|homework|schoolwork|class|teacher\s+said\s+my\s+work)\b/i.test(transcript)) {
    return 'academic_stress';
  }

  if (/\b(mom|dad|home|family|brother|sister|parents)\b/i.test(transcript)) {
    return 'family_conflict';
  }

  if (/\b(stomach|headache|sick|hurt\s+my\s+body|pain|bruise|bleeding)\b/i.test(transcript)) {
    return 'physical_complaint';
  }

  return 'emotional_regulation';
}

export function determineEscalation(
  transcript: string,
  conversationHistory: string,
  studentPatternHistory: PatternTracker[],
): EscalationType {
  if (detectHighRiskContent(transcript) || detectHighRiskContent(conversationHistory)) {
    return EscalationType.IMMEDIATE;
  }

  if (detectHelpRequest(transcript)) {
    return EscalationType.IMMEDIATE;
  }

  const concernType = getCurrentConcernType(transcript);
  const hasPattern = detectRepeatedPattern(concernType, studentPatternHistory);
  const hasIntensity = detectHighEmotionalIntensity(conversationHistory);
  const hasPowerImbalance = detectPowerImbalance(conversationHistory);

  if (hasPattern && (hasIntensity || hasPowerImbalance)) {
    return EscalationType.IMMEDIATE;
  }

  if (hasPattern) {
    return EscalationType.PATTERN;
  }

  return EscalationType.NONE;
}

function extractPrimaryEmotion(text: string): string {
  const emotionMap: Array<{ emotion: string; regex: RegExp }> = [
    { emotion: 'sad', regex: /\b(sad|down|cry|crying)\b/i },
    { emotion: 'angry', regex: /\b(angry|mad|furious|upset)\b/i },
    { emotion: 'scared', regex: /\b(scared|afraid|terrified|worried)\b/i },
    { emotion: 'lonely', regex: /\b(lonely|alone|left out|excluded)\b/i },
  ];

  const found = emotionMap.find((entry) => entry.regex.test(text));
  return found ? found.emotion : 'unspecified';
}

function compactSummary(summary: string, maxWords = 20): string {
  const words = summary.trim().split(/\s+/);
  if (words.length <= maxWords) {
    return summary.trim();
  }

  return `${words.slice(0, maxWords).join(' ')}...`;
}

export function generateTeacherAlert(response: TurtleResponse, studentConfirmedEscalation: boolean): TeacherAlert {
  const concernCategory = getCurrentConcernType(response.summary || response.listeningHealing || '');
  return {
    timestamp: new Date().toISOString(),
    escalationType: response.escalationType || EscalationType.NONE,
    urgency: response.urgency || UrgencyLevel.GREEN,
    summary: compactSummary(response.summary || 'Student requested support.'),
    concernCategory,
    primaryEmotion: extractPrimaryEmotion(response.summary || response.listeningHealing || ''),
    patternFlag: response.escalationType === EscalationType.PATTERN,
    studentConfirmedEscalation,
    actionSuggestion:
      response.urgency === UrgencyLevel.RED
        ? 'Check in with student privately'
        : 'Monitor for pattern continuation',
  };
}
