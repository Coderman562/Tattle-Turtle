export type CriticalSafetyReasonCode =
  | 'self_harm_intent_or_plan'
  | 'serious_illegal_harm_intent_or_plan';

export interface CriticalSafetyMatch {
  matched: boolean;
  reasonCode?: CriticalSafetyReasonCode;
  evidenceQuote?: string;
}

type TriggerRule = {
  reasonCode: CriticalSafetyReasonCode;
  patterns: RegExp[];
};

function normalizeText(input: string): string {
  return input
    .toLowerCase()
    .replace(/[’`]/g, "'")
    .replace(/[^a-z0-9'\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

const SELF_HARM_PATTERNS: RegExp[] = [
  /\bi\s*(?:am|'m)?\s*(?:going to|gonna|will|plan(?:ning)? to|want to|wanna|need to|might|could|should|feel like|thinking about|thought about|thought of|trying to|tried to|attempt(?:ed|ing)? to)\s*(?:kill\s*my\s*self|kill\s*myself|hurt\s*my\s*self|hurt\s*myself|harm\s*my\s*self|harm\s*myself|cut\s*my\s*self|cut\s*myself|end\s*my\s*life|take\s*my\s*life|die|commit\s*suicide|do\s*suicide|overdose|hang\s*my\s*self|hang\s*myself|jump\s*off|shoot\s*my\s*self|shoot\s*myself|poison\s*my\s*self|poison\s*myself|kms)\b/i,
  /\bi\s*(?:have|had)\s*(?:a\s*)?plan\s*(?:to|for)\s*(?:kill\s*my\s*self|kill\s*myself|hurt\s*my\s*self|hurt\s*myself|harm\s*my\s*self|harm\s*myself|end\s*my\s*life|suicide|overdose|jump\s*off|hang\s*my\s*self|hang\s*myself|kms)\b/i,
  /\bi\s*(?:tried|attempted)\s*to\s*(?:kill\s*my\s*self|kill\s*myself|hurt\s*my\s*self|hurt\s*myself|harm\s*my\s*self|harm\s*myself|end\s*my\s*life|commit\s*suicide|overdose|kms)\b/i,
  /\bi\s*(?:do\s*not|don't)\s*want\s*to\s*(?:live|be alive)\b/i,
  /\bi\s*(?:want to|wanna)\s*(?:die|be dead|stop living|not be alive)\b/i,
  /\bi\s*wish\s*i\s*(?:was|were)\s*dead\b/i,
  /\beveryone\s*would\s*be\s*better\s*off\s*without\s*me\b/i,
  /\b(?:thinking|thoughts?)\s*(?:about|of)\s*(?:suicide|killing\s*my\s*self|killing\s*myself|hurting\s*my\s*self|hurting\s*myself|harming\s*my\s*self|harming\s*myself|ending\s*my\s*life|kms)\b/i,
  /\bi\s*(?:am|'m)\s*(?:suicidal|having suicidal thoughts)\b/i,
];

const ILLEGAL_HARM_PATTERNS: RegExp[] = [
  /\b(?:i|we)\s*(?:am|'m|are|'re)?\s*(?:going to|gonna|will|plan(?:ning)? to|want to|wanna)\s*(?:stab|shoot|kill|jump|attack|bomb|burn|set\s*fire\s*to|poison|kidnap|beat\s*up|beat\s*(?:him|her|them|someone|a kid|my teacher|people|kids)\s*up)\s*(?:him|her|them|someone|a kid|my teacher|people|kids)?\b/i,
  /\b(?:i|we)\s*(?:am|'m|are|'re)?\s*(?:going to|gonna|will|plan(?:ning)? to|want to|wanna)\s*(?:bring|carry|use|take)\s*(?:a\s*)?(?:gun|knife|weapon|bomb)\s*(?:to|at)?\s*(?:school|class|campus)?\b/i,
  /\b(?:i|we)\s*(?:brought|bringing|have|got)\s*(?:a\s*)?(?:gun|knife|weapon|bomb)\b.*\b(?:shoot|stab|kill|hurt|attack|bomb)\b/i,
  /\b(?:i|we)\s*(?:am|'m|are|'re)?\s*(?:going to|gonna|will|plan(?:ning)? to|want to|wanna)\s*(?:break\s*the\s*law|break\s*the\s*constitution|do\s*something\s*illegal|commit\s*a\s*crime|rob|steal\s*from|burn\s*down|set\s*fire\s*to|sell\s*drugs|deal\s*drugs)\b/i,
];

const TRIGGER_RULES: TriggerRule[] = [
  {
    reasonCode: 'self_harm_intent_or_plan',
    patterns: SELF_HARM_PATTERNS,
  },
  {
    reasonCode: 'serious_illegal_harm_intent_or_plan',
    patterns: ILLEGAL_HARM_PATTERNS,
  },
];

export function detectCriticalTeacherRequiredEvidence(text: string): CriticalSafetyMatch {
  const source = normalizeText(text);
  if (!source) {
    return { matched: false };
  }

  for (const rule of TRIGGER_RULES) {
    for (const pattern of rule.patterns) {
      const hit = source.match(pattern);
      if (hit && hit[0]) {
        return {
          matched: true,
          reasonCode: rule.reasonCode,
          evidenceQuote: hit[0].slice(0, 220),
        };
      }
    }
  }

  return { matched: false };
}
