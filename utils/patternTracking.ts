import { TeacherAlert } from '../types';

const TEACHER_ALERTS_KEY = 'tattle_turtle_teacher_alerts_v2';

function getStorage(): Storage | null {
  if (typeof window === 'undefined' || !window.localStorage) {
    return null;
  }

  return window.localStorage;
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
