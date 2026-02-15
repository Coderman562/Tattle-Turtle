import {
  Activity,
  BookRecommendation,
  ConcernType,
  GrowthMoment,
  ParentSummary,
  ReadingMaterial,
  StudentInfo,
  TurtleConversation,
  UrgencyLevel,
} from '../types';

const CONVERSATION_STORE_KEY = 'tattle_turtle_parent_conversations_v1';
const SUMMARY_STORE_KEY = 'tattle_turtle_parent_summaries_v1';
const SENT_SUMMARY_STORE_KEY = 'tattle_turtle_parent_sent_summaries_v1';

type ConversationStore = Record<string, TurtleConversation[]>;
type SummaryStore = Record<string, ParentSummary[]>;
type SentSummaryMetaStore = Record<string, { generatedAt: string; weekKey: string }>;

function getStorage(): Storage | null {
  if (typeof window === 'undefined' || !window.localStorage) {
    return null;
  }

  return window.localStorage;
}

function getConversationStore(): ConversationStore {
  const storage = getStorage();
  if (!storage) {
    return {};
  }
  const raw = storage.getItem(CONVERSATION_STORE_KEY);
  if (!raw) {
    return {};
  }
  try {
    return JSON.parse(raw) as ConversationStore;
  } catch {
    return {};
  }
}

function saveConversationStore(store: ConversationStore): void {
  const storage = getStorage();
  if (!storage) {
    return;
  }
  storage.setItem(CONVERSATION_STORE_KEY, JSON.stringify(store));
}

function getSummaryStore(): SummaryStore {
  const storage = getStorage();
  if (!storage) {
    return {};
  }
  const raw = storage.getItem(SUMMARY_STORE_KEY);
  if (!raw) {
    return {};
  }
  try {
    return JSON.parse(raw) as SummaryStore;
  } catch {
    return {};
  }
}

function saveSummaryStore(store: SummaryStore): void {
  const storage = getStorage();
  if (!storage) {
    return;
  }
  storage.setItem(SUMMARY_STORE_KEY, JSON.stringify(store));
}

function getSentSummaryMetaStore(): SentSummaryMetaStore {
  const storage = getStorage();
  if (!storage) {
    return {};
  }
  const raw = storage.getItem(SENT_SUMMARY_STORE_KEY);
  if (!raw) {
    return {};
  }
  try {
    return JSON.parse(raw) as SentSummaryMetaStore;
  } catch {
    return {};
  }
}

function saveSentSummaryMetaStore(store: SentSummaryMetaStore): void {
  const storage = getStorage();
  if (!storage) {
    return;
  }
  storage.setItem(SENT_SUMMARY_STORE_KEY, JSON.stringify(store));
}

function concernTheme(concernType: ConcernType): string {
  switch (concernType) {
    case 'peer_conflict':
      return 'friendship navigation';
    case 'social_exclusion':
      return 'inclusion and belonging';
    case 'academic_stress':
      return 'learning confidence';
    case 'family_conflict':
      return 'home communication';
    case 'physical_complaint':
      return 'body awareness and asking for help';
    case 'emotional_regulation':
    default:
      return 'emotional regulation';
  }
}

function inLastDays(isoDate: string, days: number): boolean {
  const parsed = new Date(isoDate).getTime();
  if (!Number.isFinite(parsed)) {
    return false;
  }
  return Date.now() - parsed <= days * 24 * 60 * 60 * 1000;
}

function weekRangeLabel(now = new Date()): string {
  const end = new Date(now);
  const start = new Date(now);
  start.setDate(now.getDate() - 6);
  const toLabel = (value: Date) =>
    value.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  return `${toLabel(start)} - ${toLabel(end)}`;
}

function weekKey(now = new Date()): string {
  const start = new Date(now);
  start.setHours(0, 0, 0, 0);
  const day = start.getDay();
  const diff = (day + 6) % 7;
  start.setDate(start.getDate() - diff);
  return start.toISOString().slice(0, 10);
}

function countRegex(text: string, pattern: RegExp): number {
  const matches = text.match(pattern);
  return matches ? matches.length : 0;
}

function getTopTheme(conversations: TurtleConversation[]): string {
  const counts = new Map<string, number>();
  conversations.forEach((conversation) => {
    const theme = concernTheme(conversation.concernType);
    counts.set(theme, (counts.get(theme) || 0) + 1);
  });
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  return ranked[0]?.[0] || 'everyday emotional growth';
}

function growthLanguage(input: string): string {
  return input
    .replace(/\bstruggled with\b/gi, 'practiced navigating')
    .replace(/\bdifficulty with\b/gi, 'developing skills in')
    .replace(/\bfailed to\b/gi, 'learning to')
    .replace(/\bunable to\b/gi, 'building capacity for')
    .replace(/\bissues with\b/gi, 'growing understanding of')
    .replace(/\bconcerns about\b/gi, 'exploring')
    .replace(/\bproblems managing\b/gi, 'strengthening ability to')
    .replace(/\bpoor behavior\b/gi, 'learning appropriate behavior')
    .replace(/\bacting out\b/gi, 'expressing big feelings')
    .replace(/\bmisbehavior\b/gi, 'still developing self-regulation');
}

function urgencyMessage(urgency: UrgencyLevel, theme: string): string {
  if (urgency === UrgencyLevel.YELLOW) {
    return `Your child is developing resilience by working through ${theme} across multiple conversations.`;
  }
  if (urgency === UrgencyLevel.RED) {
    return `Your child's teacher will reach out about how to support them with ${theme}.`;
  }
  return 'Your child is building everyday emotional skills through small, brave conversations.';
}

function buildReadingMaterial(theme: string, urgency: UrgencyLevel): ReadingMaterial {
  const intro = urgencyMessage(urgency, theme);
  const quickRead = growthLanguage(
    `This week your child kept showing up to practice feelings language. That is common at this age and part of healthy development. Children build regulation and confidence by naming feelings and revisiting hard moments with a trusted listener.`
  );

  return {
    title: 'Weekly Family Connection Read',
    intro,
    quickRead,
    tips: [
      'Name one feeling you noticed in your own day, then invite your child to share one too.',
      'Celebrate effort language like “You were brave to talk about that.”',
      `Ask one short check-in about ${theme} during bedtime or dinner.`,
    ],
    parentScript:
      'I love how you practiced sharing your feelings this week. I am here to listen and learn with you. Want to tell me one moment you felt proud of yourself?',
  };
}

function buildActivity(theme: string): Activity {
  return {
    title: 'Two-Minute Feelings Replay',
    durationMinutes: 7,
    materials: ['No special materials needed'],
    steps: [
      'Ask your child to pick one feeling from today.',
      'Take three slow breaths together.',
      `Talk about one small moment related to ${theme} and one thing they did bravely.`,
    ],
    connectionQuestion: 'What helped your body feel a little calmer in that moment?',
  };
}

function buildActivities(theme: string): Activity[] {
  return [
    buildActivity(theme),
    {
      title: 'Rose, Thorn, Bud Check-In',
      durationMinutes: 8,
      materials: ['No special materials needed'],
      steps: [
        'Take turns sharing one good moment (rose), one hard moment (thorn), and one hope (bud).',
        `Tie the thorn to ${theme} and name one skill your child used.`,
        'End with one appreciation sentence for your child.',
      ],
      connectionQuestion: 'Which part felt easiest to talk about today?',
    },
    {
      title: 'Feelings Charades Mini-Game',
      durationMinutes: 6,
      materials: ['No special materials needed'],
      steps: [
        'Each person acts out a feeling with face and body only.',
        'Others guess and then name when they felt that feeling this week.',
        'Celebrate one brave share from your child.',
      ],
      connectionQuestion: 'What helps you most when that feeling shows up?',
    },
  ];
}

function buildBookRecommendations(theme: string): BookRecommendation[] {
  const books: BookRecommendation[] = [
    {
      title: 'The Invisible String',
      author: 'Patrice Karst',
      theme: 'connection and belonging',
      ratingOutOf5: 4.7,
      whyItFits: 'Supports conversations about staying connected even during hard social moments.',
    },
    {
      title: 'The Color Monster',
      author: 'Anna Llenas',
      theme: 'naming feelings',
      ratingOutOf5: 4.5,
      whyItFits: 'Helps children sort and label emotions in a concrete, visual way.',
    },
    {
      title: 'When Sophie Gets Angry - Really, Really Angry...',
      author: 'Molly Bang',
      theme: 'anger and calming',
      ratingOutOf5: 4.2,
      whyItFits: 'Normalizes big feelings and shows a child-friendly path back to calm.',
    },
    {
      title: 'Enemy Pie',
      author: 'Derek Munson',
      theme: 'friendship navigation',
      ratingOutOf5: 4.4,
      whyItFits: 'Reframes peer conflict into perspective-taking and repair.',
    },
    {
      title: 'Ruby Finds a Worry',
      author: 'Tom Percival',
      theme: 'anxiety and sharing',
      ratingOutOf5: 4.6,
      whyItFits: 'Encourages children to talk about worries before they grow.',
    },
  ];

  const prioritized = books.sort((a, b) => {
    const aBoost = (a.theme.includes(theme) || theme.includes(a.theme)) ? 1 : 0;
    const bBoost = (b.theme.includes(theme) || theme.includes(b.theme)) ? 1 : 0;
    if (aBoost !== bBoost) {
      return bBoost - aBoost;
    }
    return b.ratingOutOf5 - a.ratingOutOf5;
  });

  return prioritized.filter((book) => book.ratingOutOf5 > 3.0).slice(0, 3);
}

function buildGrowthMoment(
  conversations: TurtleConversation[],
  theme: string,
  urgency: UrgencyLevel,
  studentName: string,
): GrowthMoment {
  const combinedText = conversations.map((entry) => `${entry.studentText} ${entry.turtleSummary}`).join(' ').toLowerCase();
  const sadnessMentions = countRegex(combinedText, /\b(sad|sadness|upset|down)\b/g);
  const angerMentions = countRegex(combinedText, /\b(angry|mad|frustrated)\b/g);
  const calmingMentions = countRegex(combinedText, /\b(breathe|breath|calm|calming|pause)\b/g);
  const recurringThemeCount = conversations.filter((conversation) => concernTheme(conversation.concernType) === theme).length;

  const brightSpots = [
    `${studentName} practiced expressing emotions in ${conversations.length} conversation${conversations.length === 1 ? '' : 's'} this week.`,
    recurringThemeCount >= 2
      ? `${studentName} kept returning to ${theme}, which shows persistent practice and resilience.`
      : `${studentName} explored ${theme}, which is an important developmental skill.`,
    calmingMentions > 0
      ? `${studentName} tried a calming strategy, showing tool-building in action.`
      : `${studentName} continued learning to pause and put feelings into words.`,
  ];

  if (sadnessMentions > 0) {
    brightSpots.push(`${studentName} identified and named sadness, strengthening emotional awareness.`);
  }

  if (angerMentions > 0) {
    brightSpots.push(`${studentName} practiced talking about strong feelings instead of acting on them.`);
  }

  return {
    headline: urgencyMessage(urgency, theme),
    celebration: `${studentName} is on a unique path, and each conversation showed courage and growth.`,
    skillsPracticed: [
      'Naming feelings',
      'Asking for support',
      'Reflecting after hard moments',
    ],
    brightSpots: brightSpots.slice(0, 4),
    encouragement: 'You are already helping by staying connected and listening with warmth.',
  };
}

export function recordConversationForParentSummary(studentId: string, conversation: TurtleConversation): void {
  const store = getConversationStore();
  const existing = store[studentId] || [];
  store[studentId] = [conversation, ...existing].slice(0, 200);
  saveConversationStore(store);
}

export function getConversationsForPastDays(studentId: string, days = 7): TurtleConversation[] {
  const store = getConversationStore();
  return (store[studentId] || []).filter((entry) => inLastDays(entry.timestamp, days));
}

export function getLatestParentSummary(studentId: string): ParentSummary | null {
  const store = getSummaryStore();
  return store[studentId]?.[0] || null;
}

export function shouldGenerateWeeklySummary(studentId: string, student: StudentInfo, now = new Date()): boolean {
  if (student.optedOutOfParentCommunication || student.doNotContactParents) {
    return false;
  }

  const weeklyConversations = getConversationsForPastDays(studentId, 7);
  if (weeklyConversations.length === 0) {
    return false;
  }

  const sentStore = getSentSummaryMetaStore();
  const thisWeek = weekKey(now);
  if (sentStore[studentId]?.weekKey === thisWeek) {
    return false;
  }

  const day = now.getDay();
  const hour = now.getHours();
  const isFridayAfterFive = day === 5 && hour >= 17;
  return isFridayAfterFive;
}

export async function generateParentSummary(
  conversations: TurtleConversation[],
  student: StudentInfo,
): Promise<ParentSummary> {
  const weeklyConversations = conversations.filter((entry) => inLastDays(entry.timestamp, 7));

  if (weeklyConversations.length === 0) {
    throw new Error('No conversations available for parent summary generation.');
  }

  const parentSafeConversations = weeklyConversations.filter((entry) => entry.urgency !== UrgencyLevel.RED);
  const targetConversations = parentSafeConversations.length > 0 ? parentSafeConversations : weeklyConversations;
  const highestUrgency = targetConversations.some((entry) => entry.urgency === UrgencyLevel.YELLOW)
    ? UrgencyLevel.YELLOW
    : targetConversations.some((entry) => entry.urgency === UrgencyLevel.RED)
      ? UrgencyLevel.RED
      : UrgencyLevel.GREEN;

  const theme = getTopTheme(targetConversations);
  const studentName = student.firstName || 'Your child';

  const readingMaterial = buildReadingMaterial(theme, highestUrgency);
  const activities = buildActivities(theme);
  const activity = activities[0];
  const bookRecommendations = buildBookRecommendations(theme);
  const growthMoment = buildGrowthMoment(targetConversations, theme, highestUrgency, studentName);

  return {
    readingMaterial,
    activity,
    activities,
    bookRecommendations,
    growthMoment,
    weekCovered: weekRangeLabel(),
    generatedAt: new Date().toISOString(),
  };
}

export async function generateAndSendParentSummary(student: StudentInfo): Promise<ParentSummary | null> {
  if (student.optedOutOfParentCommunication || student.doNotContactParents) {
    return null;
  }

  const conversations = getConversationsForPastDays(student.id, 7);
  if (conversations.length === 0) {
    return null;
  }

  const summary = await generateParentSummary(conversations, student);

  const summaryStore = getSummaryStore();
  const current = summaryStore[student.id] || [];
  summaryStore[student.id] = [summary, ...current].slice(0, 20);
  saveSummaryStore(summaryStore);

  const sentStore = getSentSummaryMetaStore();
  sentStore[student.id] = {
    generatedAt: summary.generatedAt,
    weekKey: weekKey(),
  };
  saveSentSummaryMetaStore(sentStore);

  return summary;
}
