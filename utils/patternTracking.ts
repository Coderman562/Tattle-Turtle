import { ConcernType, EscalationType, PatternTracker, TeacherAlert } from '../types';

interface StudentPatternStore {
  [anonymousStudentId: string]: PatternTracker[];
}

interface StudentEscalationStore {
  [anonymousStudentId: string]: {
    lastEscalation: string;
    escalationType: EscalationType;
  };
}

const PATTERN_STORE_KEY = 'tattle_turtle_pattern_store_v1';
const ESCALATION_STORE_KEY = 'tattle_turtle_escalation_store_v1';
const TEACHER_ALERTS_KEY = 'tattle_turtle_teacher_alerts_v1';
const INTERNAL_CONCERNS_KEY = 'tattle_turtle_internal_concerns_v1';

function getStorage(): Storage | null {
  if (typeof window === 'undefined' || !window.localStorage) {
    return null;
  }

  return window.localStorage;
}

export function getPatternStore(): StudentPatternStore {
  const storage = getStorage();
  if (!storage) {
    return {};
  }

  const raw = storage.getItem(PATTERN_STORE_KEY);
  if (!raw) {
    return {};
  }

  try {
    return JSON.parse(raw) as StudentPatternStore;
  } catch {
    return {};
  }
}

export function savePatternStore(store: StudentPatternStore): void {
  const storage = getStorage();
  if (!storage) {
    return;
  }

  storage.setItem(PATTERN_STORE_KEY, JSON.stringify(store));
}

export function getStudentPatterns(studentId: string): PatternTracker[] {
  const store = getPatternStore();
  return store[studentId] || [];
}

export function updatePatternHistory(studentId: string, concernType: ConcernType, summary: string): void {
  const store = getPatternStore();

  if (!store[studentId]) {
    store[studentId] = [];
  }

  const existingPattern = store[studentId].find((pattern) => pattern.concernType === concernType);

  if (existingPattern) {
    existingPattern.occurrences.push({
      date: new Date().toISOString(),
      summary,
    });
    existingPattern.count += 1;
  } else {
    store[studentId].push({
      studentId,
      concernType,
      occurrences: [{ date: new Date().toISOString(), summary }],
      count: 1,
    });
  }

  savePatternStore(store);
}

export function getEscalationStore(): StudentEscalationStore {
  const storage = getStorage();
  if (!storage) {
    return {};
  }

  const raw = storage.getItem(ESCALATION_STORE_KEY);
  if (!raw) {
    return {};
  }

  try {
    return JSON.parse(raw) as StudentEscalationStore;
  } catch {
    return {};
  }
}

export function saveEscalationStore(store: StudentEscalationStore): void {
  const storage = getStorage();
  if (!storage) {
    return;
  }

  storage.setItem(ESCALATION_STORE_KEY, JSON.stringify(store));
}

export function hasRecentEscalation(studentId: string, hoursThreshold = 24): boolean {
  const store = getEscalationStore();
  const lastEscalation = store[studentId]?.lastEscalation;

  if (!lastEscalation) {
    return false;
  }

  const hoursSince = (Date.now() - new Date(lastEscalation).getTime()) / (1000 * 60 * 60);
  return hoursSince < hoursThreshold;
}

export function markEscalation(studentId: string, escalationType: EscalationType): void {
  const store = getEscalationStore();
  store[studentId] = {
    lastEscalation: new Date().toISOString(),
    escalationType,
  };
  saveEscalationStore(store);
}

export function saveTeacherAlert(alert: TeacherAlert): void {
  const storage = getStorage();
  if (!storage) {
    return;
  }

  const existing = storage.getItem(TEACHER_ALERTS_KEY);
  const alerts: TeacherAlert[] = existing ? (JSON.parse(existing) as TeacherAlert[]) : [];
  alerts.unshift(alert);
  storage.setItem(TEACHER_ALERTS_KEY, JSON.stringify(alerts.slice(0, 50)));
}

export function getTeacherAlerts(): TeacherAlert[] {
  const storage = getStorage();
  if (!storage) {
    return [];
  }

  const raw = storage.getItem(TEACHER_ALERTS_KEY);
  if (!raw) {
    return [];
  }

  try {
    return JSON.parse(raw) as TeacherAlert[];
  } catch {
    return [];
  }
}

export function deleteTeacherAlert(timestamp: string): void {
  const storage = getStorage();
  if (!storage) {
    return;
  }

  const alerts = getTeacherAlerts();
  const filtered = alerts.filter((alert) => alert.timestamp !== timestamp);
  storage.setItem(TEACHER_ALERTS_KEY, JSON.stringify(filtered));
}

export function logInternalConcern(studentId: string, summary: string, escalationType: EscalationType): void {
  const storage = getStorage();
  if (!storage) {
    return;
  }

  const raw = storage.getItem(INTERNAL_CONCERNS_KEY);
  const concerns: Array<{ studentId: string; summary: string; escalationType: EscalationType; timestamp: string }> =
    raw ? JSON.parse(raw) : [];

  concerns.unshift({
    studentId,
    summary,
    escalationType,
    timestamp: new Date().toISOString(),
  });

  storage.setItem(INTERNAL_CONCERNS_KEY, JSON.stringify(concerns.slice(0, 100)));
}
