import { GoogleGenAI, Type } from '@google/genai';
import { ConversationSummary, InteractionMode } from '../types';

const placeholderExercise = {
  task: 'Take one deep breath and wiggle your shoulders.',
  reward: 'Brave Breathing Badge',
};

const SUMMARY_STOP_WORDS = new Set([
  'about',
  'after',
  'again',
  'also',
  'always',
  'am',
  'and',
  'are',
  'because',
  'been',
  'being',
  'but',
  'can',
  'did',
  'does',
  'feel',
  'feeling',
  'for',
  'from',
  'had',
  'has',
  'have',
  'here',
  'how',
  'into',
  'just',
  'like',
  'maybe',
  'more',
  'need',
  'not',
  'now',
  'our',
  'really',
  'said',
  'share',
  'sharing',
  'should',
  'that',
  'the',
  'their',
  'them',
  'then',
  'there',
  'they',
  'this',
  'with',
  'would',
  'you',
  'your',
]);

function normalizeForKeywordMatch(text: string): string {
  return text
    .toLowerCase()
    .replace(/[’`]/g, "'")
    .replace(/[^a-z0-9'\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function extractContentTokens(text: string): string[] {
  const tokens = normalizeForKeywordMatch(text).split(' ').filter(Boolean);
  const unique = new Set<string>();
  for (const token of tokens) {
    if (token.length < 4 || SUMMARY_STOP_WORDS.has(token)) {
      continue;
    }
    unique.add(token);
  }
  return [...unique];
}

function parseStudentTurns(conversationHistory: string): string[] {
  const normalized = conversationHistory
    .replace(/\s*(Student|Turtle)\s*:/gi, '\n$1:')
    .trim();
  const turns = normalized.matchAll(/(?:^|\n)\s*(Student|Turtle)\s*:\s*([\s\S]*?)(?=(?:\n\s*(?:Student|Turtle)\s*:)|$)/gi);
  const studentTurns: string[] = [];

  for (const turn of turns) {
    if ((turn[1] || '').toLowerCase() !== 'student') {
      continue;
    }
    const content = (turn[2] || '').replace(/\s+/g, ' ').trim();
    if (content) {
      studentTurns.push(content);
    }
  }

  return studentTurns;
}

function extractStudentUtterances(latestStudentInput: string, conversationHistory: string): string[] {
  const fromHistory = parseStudentTurns(conversationHistory);
  const latest = latestStudentInput.replace(/\s+/g, ' ').trim();
  const merged = [...fromHistory];

  if (latest && (merged.length === 0 || merged[merged.length - 1].toLowerCase() !== latest.toLowerCase())) {
    merged.push(latest);
  }

  if (merged.length > 0) {
    return merged.slice(-8);
  }

  if (latest) {
    return [latest];
  }

  const fallback = conversationHistory.replace(/\s+/g, ' ').trim();
  return fallback ? [fallback] : [];
}

function buildStudentSnippetContext(studentUtterances: string[]): string {
  if (studentUtterances.length === 0) {
    return '(none)';
  }
  return studentUtterances
    .map((item, index) => `${index + 1}. ${compact(item, 24)}`)
    .join('\n');
}

function getConversationKeywords(studentUtterances: string[]): string[] {
  const counts = new Map<string, number>();
  for (const utterance of studentUtterances) {
    for (const token of extractContentTokens(utterance)) {
      counts.set(token, (counts.get(token) || 0) + 1);
    }
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 30)
    .map(([word]) => word);
}

function isGroundedInConversation(summary: string, studentUtterances: string[]): boolean {
  const normalizedSummary = normalizeForKeywordMatch(summary);
  if (!normalizedSummary) {
    return false;
  }
  if (studentUtterances.length === 0) {
    return true;
  }

  const keywords = getConversationKeywords(studentUtterances);
  return keywords.some((keyword) => normalizedSummary.includes(keyword));
}

function normalizeStudentCorpus(studentUtterances: string[]): string {
  return normalizeForKeywordMatch(studentUtterances.join('\n'));
}

function getVerifiedEvidenceQuotes(evidenceQuotes: string[] | undefined, studentUtterances: string[]): string[] {
  if (!Array.isArray(evidenceQuotes) || evidenceQuotes.length === 0) {
    return [];
  }

  const corpus = normalizeStudentCorpus(studentUtterances);
  const verified: string[] = [];
  for (const quote of evidenceQuotes) {
    const cleaned = quote.replace(/\s+/g, ' ').trim();
    if (!cleaned) {
      continue;
    }
    const normalizedQuote = normalizeForKeywordMatch(cleaned);
    if (normalizedQuote.length < 6) {
      continue;
    }
    if (corpus.includes(normalizedQuote)) {
      verified.push(cleaned);
    }
  }
  return verified.slice(0, 2);
}

function hasTokenOverlap(summary: string, sourceTexts: string[]): boolean {
  const summaryTokens = new Set(extractContentTokens(summary));
  if (summaryTokens.size === 0) {
    return false;
  }
  return sourceTexts.some((source) =>
    extractContentTokens(source).some((token) => summaryTokens.has(token)),
  );
}

function buildExtractiveSummaryFromEvidence(
  studentUtterances: string[],
  verifiedEvidenceQuotes: string[],
): string {
  if (verifiedEvidenceQuotes.length >= 2) {
    return compact(
      `You shared "${compact(verifiedEvidenceQuotes[0], 6)}" and "${compact(verifiedEvidenceQuotes[1], 6)}" today.`,
      20,
    );
  }
  if (verifiedEvidenceQuotes.length === 1) {
    return compact(`You shared "${compact(verifiedEvidenceQuotes[0], 10)}" today.`, 20);
  }
  return buildGroundedSummaryFallback(studentUtterances);
}

function buildGroundedSummaryFallback(studentUtterances: string[]): string {
  const latest = studentUtterances[studentUtterances.length - 1] || '';
  if (!latest) {
    return 'You shared your feelings, and we practiced naming what mattered today.';
  }
  const highlighted = compact(latest, 11);
  return compact(`You shared "${highlighted}" and practiced naming your feelings today.`, 20);
}

function compact(text: string, maxWords: number): string {
  return text
    .trim()
    .split(/\s+/)
    .slice(0, maxWords)
    .join(' ');
}

export async function getConversationSummary(
  latestStudentInput: string,
  conversationHistory: string,
  mode: InteractionMode,
): Promise<ConversationSummary> {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const studentUtterances = extractStudentUtterances(latestStudentInput, conversationHistory);
  const studentSnippetContext = buildStudentSnippetContext(studentUtterances);

  const prompt = `
You are Tattle Turtle.
Create a child-friendly conversation wrap-up for ages 6-10.

Constraints:
- No safety classification and no escalation language.
- Keep tone warm and short.
- Ground the output in actual student words from this conversation.
- Do not write generic summaries that could fit any conversation.
- tammyResponse max 15 words.
- summary max 20 words.
- summary must mention at least one concrete detail from the student snippets below.
- Include evidenceQuotes with 1-2 exact direct quotes copied from student snippets.
- evidenceQuotes must be verbatim student phrases, not paraphrases.
- If student still needs to share, set nextAction to FOLLOW_UP and provide followUpQuestion.
- If conversation feels naturally complete, set nextAction to END.
- Otherwise set nextAction to LISTEN.

Mode: ${mode}
Latest student words: """${latestStudentInput}"""
Full transcript: """${conversationHistory || latestStudentInput}"""
Student snippets to ground against:
${studentSnippetContext}

Return JSON with:
{
  "tammyResponse": "string",
  "summary": "string",
  "reflectionHelper": "string",
  "exercise": { "task": "string", "reward": "string" },
  "nextAction": "LISTEN | FOLLOW_UP | END",
  "followUpQuestion": "string (optional)",
  "evidenceQuotes": ["string"],
  "tags": ["string"]
}
  `.trim();

  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: prompt,
    config: {
      responseMimeType: 'application/json',
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          tammyResponse: { type: Type.STRING },
          summary: { type: Type.STRING },
          reflectionHelper: { type: Type.STRING },
          exercise: {
            type: Type.OBJECT,
            properties: {
              task: { type: Type.STRING },
              reward: { type: Type.STRING },
            },
            required: ['task', 'reward'],
          },
          nextAction: { type: Type.STRING },
          followUpQuestion: { type: Type.STRING },
          evidenceQuotes: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
          },
          tags: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
          },
        },
        required: [
          'tammyResponse',
          'summary',
          'reflectionHelper',
          'exercise',
          'nextAction',
          'evidenceQuotes',
        ],
      },
    },
  });

  const parsed = JSON.parse(response.text || '{}') as Partial<{
    tammyResponse: string;
    summary: string;
    reflectionHelper: string;
    exercise: { task: string; reward: string };
    nextAction: string;
    followUpQuestion: string;
    evidenceQuotes: string[];
    tags: string[];
  }>;

  const nextAction =
    parsed.nextAction === 'FOLLOW_UP'
      ? 'FOLLOW_UP'
      : parsed.nextAction === 'END'
        ? 'END'
        : 'LISTEN';

  const tammyResponse = compact(
    parsed.tammyResponse || "Thank you for sharing. I'm here with you.",
    15,
  );
  const parsedSummary = compact(parsed.summary || '', 20);
  const verifiedEvidenceQuotes = getVerifiedEvidenceQuotes(parsed.evidenceQuotes, studentUtterances);
  const summaryMatchesConversation = isGroundedInConversation(parsedSummary, studentUtterances);
  const summaryMatchesEvidence =
    verifiedEvidenceQuotes.length === 0 || hasTokenOverlap(parsedSummary, verifiedEvidenceQuotes);
  const groundedSummary =
    summaryMatchesConversation && summaryMatchesEvidence
      ? parsedSummary
      : buildExtractiveSummaryFromEvidence(studentUtterances, verifiedEvidenceQuotes);

  return {
    sufficient: nextAction !== 'FOLLOW_UP',
    shouldEndConversation: nextAction === 'END',
    closingMessage: nextAction === 'END' ? tammyResponse : undefined,
    followUpQuestion:
      nextAction === 'FOLLOW_UP'
        ? parsed.followUpQuestion || 'Can you tell me a little more about that?'
        : undefined,
    listeningHealing: tammyResponse,
    reflectionHelper: parsed.reflectionHelper || 'Thanks for sharing with me.',
    exercise: parsed.exercise || placeholderExercise,
    summary: groundedSummary,
    tammyResponse,
    tags: Array.isArray(parsed.tags) ? parsed.tags : [],
    nextAction: nextAction === 'FOLLOW_UP' ? 'FOLLOW_UP' : 'LISTEN',
  };
}
