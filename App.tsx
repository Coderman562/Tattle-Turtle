import React, { useEffect, useRef, useState } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';
import {
  ConversationEvent,
  ConversationSummary,
  EscalationType,
  InteractionMode,
  ParentSummary,
  SafetyDecision,
  SafetyOutcome,
  SessionState,
  StudentInfo,
  TeacherAlert,
  UrgencyLevel,
} from './types';
import { getConversationSummary } from './services/conversationSummaryService';
import { ConversationEventEmitter } from './services/conversationEventEmitter';
import { EvaluatorClient } from './services/evaluatorClient';
import { buildSystemInstruction } from './services/socraticPrompter';
import { detectCriticalTeacherRequiredEvidence } from './services/criticalSafetyDetector';
import {
  generateAndSendParentSummary,
  recordConversationForParentSummary,
} from './services/parentSummaryGenerator';
import { deleteTeacherAlert, getTeacherAlerts, saveTeacherAlert } from './utils/patternTracking';

function encode(bytes: Uint8Array) {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);
  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function createBlob(data: Float32Array): { data: string; mimeType: string } {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000',
  };
}

export default function App() {
  const [state, setState] = useState<SessionState>({
    step: 'WELCOME',
    interactionMode: 'LISTENING',
    response: null,
  });
  const [error, setError] = useState<string | null>(null);
  const [activeResultCard, setActiveResultCard] = useState<'SUMMARY' | null>(null);
  const [volume, setVolume] = useState(0);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [teacherContactStatus, setTeacherContactStatus] = useState<string | null>(null);
  const [showTeacherDashboard, setShowTeacherDashboard] = useState(false);
  const [teacherAlerts, setTeacherAlerts] = useState<TeacherAlert[]>([]);
  const [teacherDashboardToast, setTeacherDashboardToast] = useState<string | null>(null);
  const [liveParentSummary, setLiveParentSummary] = useState<ParentSummary | null>(null);
  const [showLiveParentSummary, setShowLiveParentSummary] = useState(false);
  const [studentNotice, setStudentNotice] = useState<string | null>(null);
  const [safetyOutcome, setSafetyOutcome] = useState<SafetyOutcome>('GREEN');
  const [latestDecisionReason, setLatestDecisionReason] = useState<string | null>(null);

  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const outputGainNodeRef = useRef<GainNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const liveSessionTokenRef = useRef(0);
  const conversationHistoryRef = useRef('');
  const sessionPromiseRef = useRef<any>(null);
  const evaluatorClientRef = useRef<EvaluatorClient | null>(null);
  const conversationEmitterRef = useRef<ConversationEventEmitter>(new ConversationEventEmitter());
  const activeConversationIdRef = useRef('');
  const studentIdRef = useRef<string>('anonymous-student');
  const studentInfoRef = useRef<StudentInfo>({ id: 'anonymous-student' });
  const isEndingSessionRef = useRef(false);
  const recentStudentTextRef = useRef('');
  const teacherRequiredThisSessionRef = useRef(false);
  const lastSeenTeacherAlertRef = useRef<string | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptRef = useRef(0);

  const silenceTimerRef = useRef<any>(null);
  const silencePromptsRef = useRef([
    "It's okay. Take your time.",
    "I'm still here if you want to share.",
    'You can start anywhere you like.',
    'Even a little bit is okay.',
  ]);
  const currentSilenceIndexRef = useRef(0);

  const latestSafetyDecisionRef = useRef<SafetyDecision | null>(null);

  const clearReconnectTimer = () => {
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  };

  const scheduleVoiceReconnect = (reason: string) => {
    if (
      isEndingSessionRef.current ||
      teacherRequiredThisSessionRef.current ||
      !activeConversationIdRef.current
    ) {
      return;
    }

    clearReconnectTimer();

    reconnectAttemptRef.current += 1;
    const attempt = reconnectAttemptRef.current;
    const delayMs = Math.min(1000 * 2 ** (attempt - 1), 8000);
    setError(`Connection hiccup (${reason}). Reconnecting...`);

    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      if (isEndingSessionRef.current || teacherRequiredThisSessionRef.current) {
        return;
      }
      void startVoiceSession("I'm back and still listening.");
    }, delayMs);
  };

  const appendConversationTurn = (role: 'Student' | 'Turtle', text: string) => {
    const cleanedText = text.replace(/\s+/g, ' ').trim();
    if (!cleanedText) {
      return;
    }
    conversationHistoryRef.current = conversationHistoryRef.current
      ? `${conversationHistoryRef.current}\n${role}: ${cleanedText}`
      : `${role}: ${cleanedText}`;
  };

  const resetSilenceTimer = () => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
    }

    silenceTimerRef.current = setTimeout(() => {
      if (state.step === 'VOICE_CHAT' && sessionPromiseRef.current && !isSpeaking) {
        const prompt = silencePromptsRef.current[currentSilenceIndexRef.current];
        currentSilenceIndexRef.current =
          (currentSilenceIndexRef.current + 1) % silencePromptsRef.current.length;

        sessionPromiseRef.current.then((session: any) => {
          session.send({
            clientContent: {
              turns: [
                {
                  role: 'user',
                  parts: [{ text: `[SYSTEM: The child is hesitant. Say exactly: "${prompt}"]` }],
                },
              ],
              turnComplete: true,
            },
          });
        });
      }
    }, 4000);
  };

  useEffect(() => {
    return () => {
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
      }
    };
  }, [state.step, isSpeaking]);

  useEffect(() => {
    return () => {
      clearReconnectTimer();
    };
  }, []);

  useEffect(() => {
    const key = 'tattle_turtle_student_id_v1';
    const existing = localStorage.getItem(key);
    if (existing) {
      studentIdRef.current = existing;
      studentInfoRef.current = { id: existing };
      return;
    }

    const generated = `student-${Math.random().toString(36).slice(2, 10)}`;
    studentIdRef.current = generated;
    studentInfoRef.current = { id: generated };
    localStorage.setItem(key, generated);
  }, []);

  useEffect(() => {
    setTeacherAlerts(getTeacherAlerts());
  }, []);

  useEffect(() => {
    if (!showTeacherDashboard) {
      return;
    }

    const refresh = () => {
      setTeacherAlerts(getTeacherAlerts());
    };

    refresh();
    const interval = window.setInterval(refresh, 1500);
    return () => window.clearInterval(interval);
  }, [showTeacherDashboard]);

  useEffect(() => {
    if (teacherAlerts.length === 0) {
      return;
    }

    const newestAlert = teacherAlerts[0];
    if (lastSeenTeacherAlertRef.current && lastSeenTeacherAlertRef.current !== newestAlert.timestamp) {
      lastSeenTeacherAlertRef.current = newestAlert.timestamp;
      setTeacherDashboardToast('New teacher alert created');
      const timeout = window.setTimeout(() => setTeacherDashboardToast(null), 3500);
      return () => window.clearTimeout(timeout);
    }

    lastSeenTeacherAlertRef.current = newestAlert.timestamp;
  }, [teacherAlerts]);

  const showDashboardToast = (message: string) => {
    setTeacherDashboardToast(message);
    window.setTimeout(() => setTeacherDashboardToast(null), 3500);
  };

  const buildGuaranteedParentSummary = (
    reason: 'CONVERSATION_END' | 'TEACHER_REQUIRED',
    cause: 'OPT_OUT' | 'NO_ELIGIBLE_CONVERSATIONS' | 'GENERATION_FAILURE',
  ): ParentSummary => {
    const now = new Date();
    const studentName = studentInfoRef.current.firstName || 'Your child';
    const parentCommsDisabled =
      Boolean(studentInfoRef.current.optedOutOfParentCommunication) ||
      Boolean(studentInfoRef.current.doNotContactParents);

    const causeLine =
      cause === 'OPT_OUT'
        ? 'Parent contact is currently disabled, so this summary is shown in-app only.'
        : cause === 'NO_ELIGIBLE_CONVERSATIONS'
          ? 'No eligible saved conversations were available, so a fallback summary was generated.'
          : 'A temporary generation issue occurred, so a fallback summary was generated.';

    const urgencyLine =
      reason === 'TEACHER_REQUIRED'
        ? `${studentName} asked for immediate trusted-adult support.`
        : `${studentName} practiced sharing feelings and communication skills.`;

    return {
      readingMaterial: {
        title: reason === 'TEACHER_REQUIRED' ? 'Immediate Support Read' : 'Connection and Calm Read',
        intro: `${urgencyLine} ${causeLine}`,
        quickRead:
          reason === 'TEACHER_REQUIRED'
            ? 'Help-seeking is a strong safety skill. A calm private check-in supports regulation and trust.'
            : 'Naming feelings and telling the story of a hard moment helps children build resilience.',
        tips: [
          'Start with warmth and appreciation for sharing.',
          'Use simple language and short reflections.',
          'Ask one gentle follow-up question and listen first.',
        ],
        parentScript:
          reason === 'TEACHER_REQUIRED'
            ? 'Thank you for asking for help. You did the right thing, and I am here with you.'
            : 'Thank you for sharing your feelings with me. I am listening and we can figure this out together.',
      },
      activity: {
        title: 'Calm and Connect',
        durationMinutes: 5,
        materials: ['No special materials needed'],
        steps: [
          'Take three slow breaths together.',
          'Name one feeling.',
          'Name one next support step.',
        ],
        connectionQuestion: 'What would help you feel better right now?',
      },
      activities: [
        {
          title: 'Calm and Connect',
          durationMinutes: 5,
          materials: ['No special materials needed'],
          steps: [
            'Take three slow breaths together.',
            'Name one feeling.',
            'Name one next support step.',
          ],
          connectionQuestion: 'What would help you feel better right now?',
        },
      ],
      bookRecommendations: [],
      growthMoment: {
        headline:
          reason === 'TEACHER_REQUIRED'
            ? `${studentName} used a brave safety skill by reaching out for support.`
            : `${studentName} practiced expressing feelings with courage.`,
        celebration:
          reason === 'TEACHER_REQUIRED'
            ? 'Asking for trusted adult help is an important protective skill.'
            : 'Each conversation builds emotional awareness and communication confidence.',
        skillsPracticed: ['Naming feelings', 'Asking for support', 'Reflecting after hard moments'],
        brightSpots: [
          parentCommsDisabled
            ? 'Summary remained available in-app even with parent communication disabled.'
            : 'Summary remained available despite generation constraints.',
          `${studentName} stayed engaged in the reflection process.`,
        ],
        encouragement: 'Consistent calm check-ins strengthen trust and regulation over time.',
      },
      weekCovered: now.toLocaleDateString(),
      generatedAt: now.toISOString(),
    };
  };

  const stopRealtimeResources = () => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
    }

    activeSourcesRef.current.forEach((source) => {
      try {
        source.onended = null;
        source.stop(0);
      } catch {
        // ignore already-stopped nodes
      }
      try {
        source.disconnect();
      } catch {
        // ignore disconnect errors
      }
    });
    activeSourcesRef.current.clear();
    setIsSpeaking(false);
    nextStartTimeRef.current = 0;

    scriptProcessorRef.current?.disconnect();
    scriptProcessorRef.current = null;

    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;
  };

  const emitEvaluationEvent = (event: ConversationEvent) => {
    if (!event.conversationId || event.conversationId !== activeConversationIdRef.current) {
      return;
    }
    evaluatorClientRef.current?.emit(event);
  };

  const forceStopVoiceSession = () => {
    isEndingSessionRef.current = true;
    clearReconnectTimer();
    reconnectAttemptRef.current = 0;
    liveSessionTokenRef.current += 1;

    if (outputGainNodeRef.current && outputAudioContextRef.current) {
      const now = outputAudioContextRef.current.currentTime;
      outputGainNodeRef.current.gain.cancelScheduledValues(now);
      outputGainNodeRef.current.gain.setValueAtTime(0, now);
    }

    stopRealtimeResources();
    outputAudioContextRef.current?.suspend().catch(() => {});
    outputAudioContextRef.current?.close().catch(() => {});
    outputAudioContextRef.current = null;
    outputGainNodeRef.current = null;

    inputAudioContextRef.current?.suspend().catch(() => {});
    inputAudioContextRef.current?.close().catch(() => {});
    inputAudioContextRef.current = null;

    sessionPromiseRef.current?.then((session: any) => session.close()).catch(() => {});
    sessionPromiseRef.current = null;

    evaluatorClientRef.current?.terminate();
    evaluatorClientRef.current = null;
    activeConversationIdRef.current = '';
  };

  const generateLiveParentSummary = async (
    reason: 'CONVERSATION_END' | 'TEACHER_REQUIRED',
    conversationOverride?: {
      studentText: string;
      turtleSummary: string;
      urgency: UrgencyLevel;
      escalationType: EscalationType;
      tags?: string[];
    },
  ) => {
    try {
      if (conversationOverride) {
        recordConversationForParentSummary(studentIdRef.current, {
          timestamp: new Date().toISOString(),
          studentText: conversationOverride.studentText,
          turtleSummary: conversationOverride.turtleSummary,
          urgency: conversationOverride.urgency,
          concernType: 'emotional_regulation',
          escalationType: conversationOverride.escalationType,
          tags: conversationOverride.tags || [],
        });
      }

      const summary = await generateAndSendParentSummary(studentInfoRef.current);
      const finalSummary =
        summary ||
        buildGuaranteedParentSummary(
          reason,
          studentInfoRef.current.optedOutOfParentCommunication || studentInfoRef.current.doNotContactParents
            ? 'OPT_OUT'
            : 'NO_ELIGIBLE_CONVERSATIONS',
        );

      setLiveParentSummary(finalSummary);
      setShowLiveParentSummary(true);
      showDashboardToast(
        reason === 'TEACHER_REQUIRED'
          ? `Parent update is now live due to teacher-required safety outcome (${finalSummary.weekCovered}).`
          : `Parent update is now live (${finalSummary.weekCovered}).`,
      );
    } catch {
      const fallback = buildGuaranteedParentSummary(reason, 'GENERATION_FAILURE');
      setLiveParentSummary(fallback);
      setShowLiveParentSummary(true);
      showDashboardToast(
        reason === 'TEACHER_REQUIRED'
          ? `Parent update fallback is now live due to teacher-required safety outcome (${fallback.weekCovered}).`
          : `Parent update fallback is now live (${fallback.weekCovered}).`,
      );
    }
  };

  const triggerTeacherRequiredAlert = (summary: string, reasonCode: string) => {
    if (teacherRequiredThisSessionRef.current) {
      return;
    }

    teacherRequiredThisSessionRef.current = true;
    setSafetyOutcome('TEACHER_REQUIRED');

    forceStopVoiceSession();

    const alert: TeacherAlert = {
      timestamp: new Date().toISOString(),
      safetyOutcome: 'TEACHER_REQUIRED',
      summary: summary.split(/\s+/).slice(0, 24).join(' '),
      reasonCode,
      teacherNotice: summary,
      actionSuggestion: 'Check in with student privately',
    };

    saveTeacherAlert(alert);
    setTeacherAlerts(getTeacherAlerts());

    setTeacherContactStatus('Teacher has been notified to check in privately.');
    setStudentNotice('Teacher has been notified to check in privately.');

    const emergencyResponse: ConversationSummary = {
      sufficient: true,
      shouldEndConversation: true,
      closingMessage: 'I heard a big safety concern. A teacher is coming to help now.',
      listeningHealing: 'A teacher is coming to help now.',
      reflectionHelper: '',
      exercise: { task: 'Stay where you are and take slow breaths.', reward: 'Safety First' },
      summary: 'Teacher support requested now.',
      tammyResponse: 'A teacher is coming to help now.',
      nextAction: 'LISTEN',
    };

    setState((prev) => ({
      ...prev,
      step: 'RESULTS',
      response: emergencyResponse,
    }));

    void generateLiveParentSummary('TEACHER_REQUIRED', {
      studentText: conversationHistoryRef.current || recentStudentTextRef.current || summary,
      turtleSummary: summary,
      urgency: UrgencyLevel.RED,
      escalationType: EscalationType.IMMEDIATE,
      tags: ['teacher-required'],
    });
  };

  const endVoiceSession = async () => {
    if (isEndingSessionRef.current) {
      return;
    }

    clearReconnectTimer();
    reconnectAttemptRef.current = 0;
    emitEvaluationEvent(conversationEmitterRef.current.endSession(studentIdRef.current));
    isEndingSessionRef.current = true;
    stopRealtimeResources();
    setState((prev) => ({ ...prev, step: 'PROCESSING' }));

    await sessionPromiseRef.current?.then((session: any) => session.close()).catch(() => {});
    sessionPromiseRef.current = null;

    evaluatorClientRef.current?.terminate();
    evaluatorClientRef.current = null;

    await handleFinalSubmit(conversationHistoryRef.current || '(Silence)');
  };

  const applySafetyDecision = (decision: SafetyDecision) => {
    if (decision.conversationId !== activeConversationIdRef.current) {
      return;
    }

    latestSafetyDecisionRef.current = decision;
    setLatestDecisionReason(decision.reasonCode);

    console.info('[SafetyDecision]', {
      conversationId: decision.conversationId,
      utteranceId: decision.utteranceId,
      safetyOutcome: decision.safetyOutcome,
      reasonCode: decision.reasonCode,
      confidence: decision.confidence,
      latencyMs: decision.latencyMs,
    });

    if (decision.studentNotice) {
      setStudentNotice(decision.studentNotice);
    }

    if (decision.safetyOutcome === 'TEACHER_REQUIRED' || decision.teacherNotifyNow) {
      triggerTeacherRequiredAlert(
        decision.teacherNotice || 'Immediate teacher support required.',
        decision.reasonCode || 'teacher_required',
      );
      return;
    }

    setSafetyOutcome('GREEN');

    if (decision.shouldEndConversation && !isEndingSessionRef.current) {
      void endVoiceSession();
    }
  };

  const startVoiceSession = async (customGreeting?: string) => {
    setError(null);
    isEndingSessionRef.current = false;
    teacherRequiredThisSessionRef.current = false;
    setTeacherContactStatus(null);
    setStudentNotice(null);
    setSafetyOutcome('GREEN');
    setLatestDecisionReason(null);

    try {
      const startEvent = conversationEmitterRef.current.startSession(studentIdRef.current);
      activeConversationIdRef.current = startEvent.conversationId;

      evaluatorClientRef.current?.terminate();
      evaluatorClientRef.current = new EvaluatorClient(
        (decision) => applySafetyDecision(decision),
        (reason) => {
          console.warn('[SafetyEvaluatorFallback]', reason);
          showDashboardToast(`Evaluator fallback engaged: ${reason}`);
        },
      );

      const evaluatorReady = evaluatorClientRef.current.init();
      if (evaluatorReady) {
        emitEvaluationEvent(startEvent);
      } else {
        console.warn('[SafetyEvaluatorOffline] continuing voice chat without evaluator');
        showDashboardToast('Safety evaluator offline. Voice chat continues.');
      }

      const sessionToken = liveSessionTokenRef.current + 1;
      liveSessionTokenRef.current = sessionToken;

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = stream;

      setState((prev) => ({ ...prev, step: 'VOICE_CHAT' }));

      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000,
      });
      outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 24000,
      });
      outputGainNodeRef.current = outputAudioContextRef.current.createGain();
      outputGainNodeRef.current.gain.value = 1;
      outputGainNodeRef.current.connect(outputAudioContextRef.current.destination);
      nextStartTimeRef.current = 0;

      const greeting = customGreeting || 'Hey friend! Want to tell me something?';
      const systemInstruction = buildSystemInstruction({
        mode: state.interactionMode,
        greeting,
      });

      sessionPromiseRef.current = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-12-2025',
        callbacks: {
          onopen: () => {
            if (sessionToken !== liveSessionTokenRef.current || isEndingSessionRef.current) {
              return;
            }
            clearReconnectTimer();
            reconnectAttemptRef.current = 0;
            setError(null);

            sessionPromiseRef.current.then((session: any) => {
              session.send({
                clientContent: {
                  turns: [
                    {
                      role: 'user',
                      parts: [{ text: `[START SESSION: Greet the child immediately with: "${greeting}"]` }],
                    },
                  ],
                  turnComplete: true,
                },
              });
            });

            const source = inputAudioContextRef.current!.createMediaStreamSource(stream);
            const scriptProcessor = inputAudioContextRef.current!.createScriptProcessor(4096, 1, 1);
            scriptProcessorRef.current = scriptProcessor;

            scriptProcessor.onaudioprocess = (e) => {
              if (isEndingSessionRef.current) {
                return;
              }

              const inputData = e.inputBuffer.getChannelData(0);
              let sum = 0;
              let peak = 0;
              for (let i = 0; i < inputData.length; i++) {
                sum += inputData[i] * inputData[i];
                if (Math.abs(inputData[i]) > peak) {
                  peak = Math.abs(inputData[i]);
                }
              }
              const rms = Math.sqrt(sum / inputData.length);
              setVolume(Math.min(100, rms * 500));

              if (peak > 0.08) {
                resetSilenceTimer();
              }

              const pcmBlob = createBlob(inputData);
              sessionPromiseRef.current.then((session: any) => {
                session.sendRealtimeInput({ media: pcmBlob });
              });
            };

            source.connect(scriptProcessor);
            scriptProcessor.connect(inputAudioContextRef.current!.destination);
            resetSilenceTimer();
          },
          onmessage: async (message: LiveServerMessage) => {
            if (sessionToken !== liveSessionTokenRef.current || isEndingSessionRef.current) {
              return;
            }

            if (message.serverContent?.inputTranscription) {
              const childText = message.serverContent.inputTranscription.text;
              appendConversationTurn('Student', childText);
              recentStudentTextRef.current = `${recentStudentTextRef.current} ${childText}`.slice(-1000);
              resetSilenceTimer();

              const criticalMatch = detectCriticalTeacherRequiredEvidence(childText);
              if (criticalMatch.matched && criticalMatch.reasonCode) {
                triggerTeacherRequiredAlert(
                  criticalMatch.evidenceQuote
                    ? `Critical safety statement detected: "${criticalMatch.evidenceQuote}".`
                    : 'Critical safety statement detected.',
                  criticalMatch.reasonCode,
                );
                return;
              }

              emitEvaluationEvent(
                conversationEmitterRef.current.studentUtterance(studentIdRef.current, childText),
              );
            }

            if (message.serverContent?.outputTranscription) {
              const turtleText = message.serverContent.outputTranscription.text;
              appendConversationTurn('Turtle', turtleText);
              resetSilenceTimer();

              emitEvaluationEvent(
                conversationEmitterRef.current.modelUtterance(studentIdRef.current, turtleText),
              );
            }

            const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (base64Audio) {
              if (sessionToken !== liveSessionTokenRef.current || isEndingSessionRef.current) {
                return;
              }

              setIsSpeaking(true);
              const ctx = outputAudioContextRef.current;
              if (!ctx) {
                return;
              }

              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
              const buffer = await decodeAudioData(decode(base64Audio), ctx, 24000, 1);
              if (sessionToken !== liveSessionTokenRef.current || isEndingSessionRef.current) {
                return;
              }

              const source = ctx.createBufferSource();
              source.buffer = buffer;
              const outputNode = outputGainNodeRef.current || ctx.destination;
              source.connect(outputNode);
              activeSourcesRef.current.add(source);
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += buffer.duration;

              source.onended = () => {
                activeSourcesRef.current.delete(source);
                if (activeSourcesRef.current.size === 0) {
                  setIsSpeaking(false);
                  resetSilenceTimer();
                }
              };
            }
          },
          onerror: (e: ErrorEvent) => {
            if (sessionToken !== liveSessionTokenRef.current) {
              return;
            }
            console.error('Turtle API Error:', e);
            stopRealtimeResources();
            sessionPromiseRef.current?.then((session: any) => session.close()).catch(() => {});
            sessionPromiseRef.current = null;
            scheduleVoiceReconnect('live_error');
          },
          onclose: () => {
            if (
              sessionToken !== liveSessionTokenRef.current ||
              isEndingSessionRef.current ||
              teacherRequiredThisSessionRef.current
            ) {
              return;
            }
            console.debug('Session closed');
            scheduleVoiceReconnect('live_close');
          },
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } },
          systemInstruction,
          inputAudioTranscription: {},
          outputAudioTranscription: {},
        },
      });
    } catch (error: any) {
      const errorName = String(error?.name || '');
      const isMicPermissionIssue =
        errorName === 'NotAllowedError' ||
        errorName === 'NotFoundError' ||
        errorName === 'NotReadableError' ||
        errorName === 'OverconstrainedError';

      if (isMicPermissionIssue) {
        clearReconnectTimer();
        reconnectAttemptRef.current = 0;
        setError("Turtle needs microphone access. Please enable your mic and try again.");
        setState((prev) => ({ ...prev, step: 'WELCOME' }));
        return;
      }

      setState((prev) => ({ ...prev, step: 'VOICE_CHAT' }));
      scheduleVoiceReconnect('startup_error');
    }
  };

  const handleFinalSubmit = async (text: string) => {
    setState((prev) => ({ ...prev, step: 'PROCESSING' }));
    const transcriptForSummary = conversationHistoryRef.current || text;
    const studentInputForSummary = recentStudentTextRef.current.trim() || text;

    try {
      const criticalMatch = detectCriticalTeacherRequiredEvidence(transcriptForSummary);
      if (criticalMatch.matched && criticalMatch.reasonCode) {
        triggerTeacherRequiredAlert(
          criticalMatch.evidenceQuote
            ? `Critical safety statement detected: "${criticalMatch.evidenceQuote}".`
            : 'Critical safety statement detected.',
          criticalMatch.reasonCode,
        );
        return;
      }

      const response = await getConversationSummary(
        studentInputForSummary,
        transcriptForSummary,
        state.interactionMode,
      );

      const decision = latestSafetyDecisionRef.current;
      const finalOutcome = decision?.safetyOutcome || 'GREEN';
      setSafetyOutcome(finalOutcome);

      recordConversationForParentSummary(studentIdRef.current, {
        timestamp: new Date().toISOString(),
        studentText: transcriptForSummary,
        turtleSummary: response.summary || response.listeningHealing || '',
        urgency: finalOutcome === 'TEACHER_REQUIRED' ? UrgencyLevel.RED : UrgencyLevel.GREEN,
        concernType: 'emotional_regulation',
        escalationType:
          finalOutcome === 'TEACHER_REQUIRED' ? EscalationType.IMMEDIATE : EscalationType.NONE,
        tags: response.tags || [],
      });

      await generateLiveParentSummary('CONVERSATION_END');

      if (finalOutcome === 'TEACHER_REQUIRED') {
        setTeacherContactStatus('Teacher has been notified to check in privately.');
      } else {
        setTeacherContactStatus(null);
      }

      setState((prev) => ({ ...prev, step: 'RESULTS', response }));
    } catch (error) {
      console.warn('[ConversationSummaryFallback]', error);

      const summarySnippet = studentInputForSummary
        .replace(/\s+/g, ' ')
        .trim()
        .split(' ')
        .slice(0, 16)
        .join(' ');

      const fallbackResponse: ConversationSummary = {
        sufficient: true,
        shouldEndConversation: true,
        closingMessage: "Thanks for sharing. I'm still here with you.",
        listeningHealing: "I'm still here with you.",
        reflectionHelper: 'You shared something important today.',
        exercise: {
          task: 'Take one slow breath and relax your shoulders.',
          reward: 'Steady Turtle',
        },
        summary: summarySnippet
          ? `You shared "${summarySnippet}" and we stayed with that together.`
          : 'You shared your thoughts, and we stayed with them together.',
        tammyResponse: "Thanks for sharing. I'm still here with you.",
        nextAction: 'LISTEN',
      };

      const decision = latestSafetyDecisionRef.current;
      const finalOutcome = decision?.safetyOutcome || 'GREEN';
      setSafetyOutcome(finalOutcome);
      setTeacherContactStatus(
        finalOutcome === 'TEACHER_REQUIRED'
          ? 'Teacher has been notified to check in privately.'
          : null,
      );
      setError(null);

      recordConversationForParentSummary(studentIdRef.current, {
        timestamp: new Date().toISOString(),
        studentText: transcriptForSummary,
        turtleSummary: fallbackResponse.summary,
        urgency: finalOutcome === 'TEACHER_REQUIRED' ? UrgencyLevel.RED : UrgencyLevel.GREEN,
        concernType: 'emotional_regulation',
        escalationType:
          finalOutcome === 'TEACHER_REQUIRED' ? EscalationType.IMMEDIATE : EscalationType.NONE,
        tags: ['summary-fallback'],
      });

      void generateLiveParentSummary('CONVERSATION_END');

      setState((prev) => ({ ...prev, step: 'RESULTS', response: fallbackResponse }));
    }
  };

  const generateSummaryNow = async () => {
    try {
      const summary = await generateAndSendParentSummary(studentInfoRef.current);
      const finalSummary =
        summary ||
        buildGuaranteedParentSummary(
          'CONVERSATION_END',
          studentInfoRef.current.optedOutOfParentCommunication || studentInfoRef.current.doNotContactParents
            ? 'OPT_OUT'
            : 'NO_ELIGIBLE_CONVERSATIONS',
        );
      setLiveParentSummary(finalSummary);
      setShowLiveParentSummary(true);
      showDashboardToast(`Parent update is live for ${finalSummary.weekCovered}.`);
    } catch {
      const fallback = buildGuaranteedParentSummary('CONVERSATION_END', 'GENERATION_FAILURE');
      setLiveParentSummary(fallback);
      setShowLiveParentSummary(true);
      showDashboardToast(`Parent update fallback is live for ${fallback.weekCovered}.`);
    }
  };

  const resetSession = () => {
    isEndingSessionRef.current = false;
    clearReconnectTimer();
    reconnectAttemptRef.current = 0;
    teacherRequiredThisSessionRef.current = false;
    recentStudentTextRef.current = '';
    latestSafetyDecisionRef.current = null;

    stopRealtimeResources();
    sessionPromiseRef.current?.then((session: any) => session.close()).catch(() => {});
    sessionPromiseRef.current = null;

    evaluatorClientRef.current?.terminate();
    evaluatorClientRef.current = null;
    activeConversationIdRef.current = '';

    setState({
      step: 'WELCOME',
      interactionMode: 'LISTENING',
      response: null,
    });
    setError(null);
    setActiveResultCard(null);
    setTeacherContactStatus(null);
    setStudentNotice(null);
    setShowLiveParentSummary(false);
    setSafetyOutcome('GREEN');
    setLatestDecisionReason(null);
    conversationHistoryRef.current = '';
  };

  return (
    <div className="min-h-screen px-4 py-10 md:px-12 md:py-16 flex flex-col items-center">
      {showTeacherDashboard && teacherDashboardToast && (
        <div className="fixed top-6 right-6 z-50 bg-[var(--soft-coral)] text-white px-6 py-4 rounded-[1.25rem] shadow-2xl border-2 border-white">
          <p className="text-xl font-bubble">{teacherDashboardToast}</p>
        </div>
      )}

      {showLiveParentSummary && liveParentSummary && (
        <div className="fixed bottom-6 right-6 z-50 max-w-xl w-[calc(100%-3rem)] sm:w-[34rem] bg-white border-4 border-[var(--sky-blue)] rounded-[1.5rem] shadow-2xl p-5">
          <div className="flex items-start justify-between gap-4 mb-2">
            <h3 className="text-2xl font-bubble text-[var(--text-cocoa)]">Parent Update Live</h3>
            <button
              onClick={() => setShowLiveParentSummary(false)}
              className="bubbly-button bg-[var(--soft-coral)] text-white text-base px-3 py-1"
            >
              Close
            </button>
          </div>
          <p className="text-sm text-[var(--text-clay)] mb-2">Week: {liveParentSummary.weekCovered}</p>
          <div className="rounded-xl bg-[var(--mint-calm)]/30 p-3 mb-2">
            <p className="text-sm font-semibold text-[var(--text-cocoa)] mb-1">📚 Reading Materials</p>
            <p className="text-sm text-[var(--text-clay)]">{liveParentSummary.readingMaterial.title}</p>
          </div>
          <div className="rounded-xl bg-[var(--warm-butter)]/20 p-3 mb-2">
            <p className="text-sm font-semibold text-[var(--text-cocoa)] mb-1">🧩 Activities & Book Recommendations</p>
            <p className="text-sm text-[var(--text-clay)]">
              {liveParentSummary.activities.slice(0, 2).map((item) => item.title).join(' • ')}
            </p>
          </div>
          <div className="rounded-xl bg-[var(--gentle-leaf)]/20 p-3">
            <p className="text-sm font-semibold text-[var(--text-cocoa)] mb-1">🌱 Growth Moments Report</p>
            <p className="text-sm text-[var(--text-clay)]">{liveParentSummary.growthMoment.headline}</p>
          </div>
        </div>
      )}

      <header className="flex flex-col items-center mb-12 text-center">
        <div className="flex items-center gap-6 mb-2 cursor-pointer" onClick={resetSession}>
          <div className="text-8xl turtle-bounce">🐢</div>
          <h1 className="text-6xl font-bubble text-[var(--text-cocoa)] tracking-wide drop-shadow-sm">
            Tattle Turtle
          </h1>
        </div>
        <div className="flex gap-3 mb-4">
          <button
            onClick={() => setShowTeacherDashboard(false)}
            className={`bubbly-button px-6 py-3 text-xl font-bubble ${
              !showTeacherDashboard ? 'bg-[var(--sky-blue)] text-white' : 'bg-white text-[var(--text-cocoa)]'
            }`}
          >
            Student View
          </button>
          <button
            onClick={() => setShowTeacherDashboard(true)}
            className={`bubbly-button px-6 py-3 text-xl font-bubble ${
              showTeacherDashboard ? 'bg-[var(--soft-coral)] text-white' : 'bg-white text-[var(--text-cocoa)]'
            }`}
          >
            Teacher Dashboard
          </button>
        </div>
        <p className="text-2xl text-[var(--text-clay)] font-bubble italic">
          Slow and steady, we share our worries.
        </p>
      </header>

      <main className="w-full max-w-4xl flex flex-col items-center">
        {showTeacherDashboard && (
          <div className="w-full max-w-4xl bubbly-card bg-white p-8 md:p-10">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
              <h2 className="text-4xl font-bubble text-[var(--text-cocoa)]">Teacher Alerts</h2>
              <button
                onClick={() => generateSummaryNow()}
                className="bubbly-button bg-[var(--gentle-leaf)] text-white text-xl px-6 py-3"
              >
                ✓ Send Parent Update
              </button>
            </div>
            {teacherAlerts.length === 0 ? (
              <p className="text-2xl text-[var(--text-clay)]">No alerts yet.</p>
            ) : (
              <div className="flex flex-col gap-4">
                {teacherAlerts.map((alert, index) => (
                  <div key={`${alert.timestamp}-${index}`} className="rounded-[1.5rem] p-5 border-2 border-[var(--mint-calm)]">
                    <div className="flex justify-between items-start gap-4">
                      <div className="flex flex-wrap gap-3 text-lg text-[var(--text-cocoa)] mb-2">
                        <span className="font-bold">{new Date(alert.timestamp).toLocaleString()}</span>
                        <span
                          className={`px-3 py-1 rounded-full text-white ${
                            alert.safetyOutcome === 'TEACHER_REQUIRED'
                              ? 'bg-[var(--soft-coral)]'
                              : 'bg-[var(--gentle-leaf)]'
                          }`}
                        >
                          {alert.safetyOutcome}
                        </span>
                        <span className="px-3 py-1 rounded-full bg-[var(--mint-calm)]">{alert.reasonCode}</span>
                      </div>
                      <button
                        onClick={() => {
                          deleteTeacherAlert(alert.timestamp);
                          setTeacherAlerts(getTeacherAlerts());
                        }}
                        className="bubbly-button bg-[var(--gentle-leaf)] text-white text-xl px-4 py-2"
                        title="Mark handled and remove alert"
                      >
                        ✓
                      </button>
                    </div>
                    <p className="text-xl text-[var(--text-cocoa)] mb-2">{alert.summary}</p>
                    {alert.teacherNotice && (
                      <p className="text-base text-[var(--text-clay)] mb-1">Teacher note: {alert.teacherNotice}</p>
                    )}
                    {alert.actionSuggestion && (
                      <p className="text-base text-[var(--text-clay)]">Action: {alert.actionSuggestion}</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {!showTeacherDashboard && (
          <>
            {state.step === 'WELCOME' && !state.response && (
              <div className="flex flex-col items-center gap-12 w-full animate-in fade-in zoom-in duration-500">
                <div className="bg-[var(--mint-calm)] p-3 rounded-[3.5rem] shadow-sm flex gap-3 w-full max-w-xl">
                  <button
                    onClick={() => setState((s) => ({ ...s, interactionMode: 'LISTENING' }))}
                    className={`flex-1 py-5 px-6 rounded-[3rem] font-bubble text-2xl transition-all duration-300 ${
                      state.interactionMode === 'LISTENING'
                        ? 'bg-[var(--sky-blue)] text-white shadow-md'
                        : 'text-[var(--text-cocoa)] hover:bg-white/30'
                    }`}
                  >
                    👂 Ears On
                  </button>
                  <button
                    onClick={() => setState((s) => ({ ...s, interactionMode: 'SOCRATIC' }))}
                    className={`flex-1 py-5 px-6 rounded-[3rem] font-bubble text-2xl transition-all duration-300 ${
                      state.interactionMode === 'SOCRATIC'
                        ? 'bg-[var(--lavender-wonder)] text-white shadow-md'
                        : 'text-[var(--text-cocoa)] hover:bg-white/30'
                    }`}
                  >
                    🤔 Wonder
                  </button>
                </div>

                <button
                  onClick={() => startVoiceSession()}
                  className="bubbly-button bg-[var(--sky-blue)] hover:bg-[var(--sky-blue)] hover:opacity-90 text-white text-6xl font-bubble py-12 px-32 flex items-center gap-6 shadow-xl transition-all transform hover:scale-105 active:scale-95"
                >
                  <span className="text-7xl">🎤</span> Speak
                </button>

                {error && (
                  <p className="text-[var(--soft-coral)] font-bold bg-white px-10 py-4 rounded-full shadow-md text-xl">
                    {error}
                  </p>
                )}
              </div>
            )}

            {state.step === 'VOICE_CHAT' && (
              <div className="flex flex-col items-center gap-10 animate-in slide-in-from-bottom-8 duration-500">
                {studentNotice && (
                  <div className="bubbly-card bg-[var(--warm-butter)] px-6 py-4 text-center max-w-2xl">
                    <p className="text-xl font-bubble text-[var(--text-cocoa)]">{studentNotice}</p>
                  </div>
                )}

                <div className="relative group">
                  <div
                    className={`absolute inset-0 rounded-full blur-3xl opacity-40 transition-all duration-300 ${
                      isSpeaking ? 'bg-[var(--warm-sunshine)] scale-125' : 'bg-[var(--turtle-green)] scale-100'
                    }`}
                    style={{ transform: `scale(${1 + volume / 100})` }}
                  />
                  <div
                    className={`w-64 h-64 rounded-full relative z-10 ${
                      state.interactionMode === 'LISTENING' ? 'bg-[var(--mint-calm)]' : 'bg-[var(--lavender-wonder)]'
                    } flex items-center justify-center ${
                      !isSpeaking ? 'listening-pulse' : ''
                    } shadow-2xl transition-all duration-500 ${isSpeaking ? 'scale-110' : ''}`}
                  >
                    <span className={`text-[12rem] transition-all duration-300 ${isSpeaking ? 'rotate-3' : 'rotate-0'}`}>
                      🐢
                    </span>
                  </div>
                </div>

                <div className="text-center">
                  <p className="text-4xl font-bubble text-[var(--text-cocoa)] mb-2">
                    {isSpeaking ? 'Turtle is talking...' : 'Turtle is Listening...'}
                  </p>
                  <div className="flex gap-1 justify-center h-8 items-center">
                    {[1, 2, 3, 4, 5].map((i) => (
                      <div
                        key={i}
                        className={`w-2 rounded-full transition-all duration-75 ${
                          isSpeaking ? 'bg-[var(--warm-sunshine)]' : 'bg-[var(--sky-blue)]'
                        }`}
                        style={{
                          opacity: volume > i * 15 || isSpeaking ? 0.8 : 0.2,
                          height:
                            volume > i * 15 || (isSpeaking && Math.random() > 0.5) ? '2.5rem' : '0.75rem',
                        }}
                      />
                    ))}
                  </div>
                </div>

                <button
                  onClick={endVoiceSession}
                  className="bubbly-button bg-[var(--gentle-leaf)] hover:opacity-90 text-white text-5xl font-bubble py-10 px-24 shadow-lg transform transition-hover active:scale-95"
                >
                  I'm All Done!
                </button>

              </div>
            )}

            {state.step === 'PROCESSING' && (
              <div className="flex flex-col items-center gap-8 py-24 animate-in fade-in duration-500">
                <div className="text-[10rem] animate-spin text-[var(--warm-sunshine)]">🐚</div>
                <p className="text-5xl font-bubble text-[var(--text-cocoa)]">Opening my shell...</p>
              </div>
            )}

            {state.step === 'RESULTS' && state.response && (
              <div className="flex flex-col items-center gap-10 w-full animate-in fade-in duration-1000">
                <div className="text-center mb-6">
                  <p className="text-4xl font-bubble text-[var(--peachy-comfort)] italic mb-3">
                    {safetyOutcome === 'TEACHER_REQUIRED'
                      ? 'A teacher is on the way to help. ❤️'
                      : state.interactionMode === 'LISTENING'
                        ? 'I heard every bit! ❤️'
                        : 'We did some great thinking! 🧠'}
                  </p>
                  <p className="text-2xl text-[var(--text-clay)] font-medium">Here is a summary of the conversation</p>
                </div>

                {liveParentSummary && (
                  <div className="w-full max-w-3xl bg-white bubbly-card p-8 border-4 border-[var(--sky-blue)]">
                    <h3 className="text-4xl font-bubble text-[var(--text-cocoa)] mb-2">Parent Summary Dashboard</h3>
                    <p className="text-base text-[var(--text-clay)] mb-3">Week: {liveParentSummary.weekCovered}</p>
                    <div className="rounded-[1.25rem] border-2 border-[var(--sky-blue)] p-5 mb-4 bg-[var(--mint-calm)]/25">
                      <h4 className="text-2xl font-bubble text-[var(--text-cocoa)] mb-2">📚 Reading Materials</h4>
                      <p className="text-lg text-[var(--text-cocoa)] mb-2">{liveParentSummary.readingMaterial.title}</p>
                      <p className="text-base text-[var(--text-clay)] mb-2">{liveParentSummary.readingMaterial.intro}</p>
                      <p className="text-base text-[var(--text-clay)] mb-2">{liveParentSummary.readingMaterial.quickRead}</p>
                      <p className="text-base text-[var(--text-clay)]">
                        Script: "{liveParentSummary.readingMaterial.parentScript}"
                      </p>
                    </div>
                    <div className="rounded-[1.25rem] border-2 border-[var(--warm-butter)] p-5 mb-4 bg-[var(--warm-butter)]/20">
                      <h4 className="text-2xl font-bubble text-[var(--text-cocoa)] mb-2">🧩 Activities & Book Recommendations</h4>
                      <p className="text-base text-[var(--text-clay)] mb-1">
                        {liveParentSummary.activities
                          .slice(0, 3)
                          .map((item) => `${item.title} (${item.durationMinutes} min)`)
                          .join(' • ')}
                      </p>
                      <p className="text-base text-[var(--text-clay)]">
                        {liveParentSummary.bookRecommendations
                          .map((book) => `${book.title} by ${book.author} (${book.ratingOutOf5.toFixed(1)}/5)`)
                          .join(' • ')}
                      </p>
                    </div>
                    <div className="rounded-[1.25rem] border-2 border-[var(--gentle-leaf)] p-5 bg-[var(--gentle-leaf)]/20">
                      <h4 className="text-2xl font-bubble text-[var(--text-cocoa)] mb-2">🌱 Growth Moments Report</h4>
                      <p className="text-xl text-[var(--text-cocoa)] mb-2">{liveParentSummary.growthMoment.headline}</p>
                      <p className="text-base text-[var(--text-clay)] mb-2">{liveParentSummary.growthMoment.celebration}</p>
                      <p className="text-base text-[var(--text-clay)]">
                        {liveParentSummary.growthMoment.brightSpots.join(' • ')}
                      </p>
                    </div>
                  </div>
                )}

                {safetyOutcome === 'TEACHER_REQUIRED' ? (
                  <div className="bg-[var(--soft-coral)] bubbly-card p-10 text-center w-full max-w-2xl">
                    <p className="text-4xl font-bubble text-white">
                      {state.response.closingMessage || 'A teacher has been notified and is coming to help now.'}
                    </p>
                    {latestDecisionReason && (
                      <p className="text-lg text-white/90 mt-4">Decision code: {latestDecisionReason}</p>
                    )}
                  </div>
                ) : (
                  <>
                    <div className="flex w-full max-w-3xl">
                      <button
                        onClick={() => setActiveResultCard(activeResultCard === 'SUMMARY' ? null : 'SUMMARY')}
                        className={`w-full bubbly-button text-3xl font-bubble py-10 px-8 flex flex-col items-center gap-4 transition-all duration-300 ${
                          activeResultCard === 'SUMMARY'
                            ? 'bg-[var(--soft-coral)] scale-105 shadow-xl'
                            : 'bg-[var(--soft-coral)] opacity-70 hover:opacity-100'
                        } text-white`}
                      >
                        <span>Conversation Summary 💬</span>
                      </button>
                    </div>

                    <div className="w-full mt-6 min-h-[350px] flex justify-center transition-all duration-500">
                      {activeResultCard === 'SUMMARY' && (
                        <div className="bubbly-card w-full max-w-2xl bg-[var(--mint-calm)] p-12 animate-in slide-in-from-top-6 duration-500 text-center shadow-inner">
                          <h3 className="text-5xl font-bubble text-[var(--text-cocoa)] mb-8">Conversation Summary 💬</h3>
                          <p className="text-3xl text-[var(--text-clay)] font-bubble italic leading-relaxed">
                            "{state.response.summary}"
                          </p>
                        </div>
                      )}
                    </div>
                  </>
                )}

                {teacherContactStatus && (
                  <div className="bg-white bubbly-card p-6 w-full max-w-2xl border-4 border-[var(--gentle-leaf)]">
                    <p className="text-2xl font-bubble text-[var(--text-cocoa)] text-center">{teacherContactStatus}</p>
                  </div>
                )}

                <div className="mt-12 flex flex-col sm:flex-row gap-4">
                  <button
                    onClick={resetSession}
                    className="bubbly-button bg-[var(--mint-calm)] text-[var(--text-cocoa)] text-3xl font-bubble py-5 px-16 transition-all hover:opacity-80"
                  >
                    Start New Story
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </main>

      <footer className="mt-24 text-center text-[var(--text-clay)] font-bubble text-3xl italic opacity-50 pb-16">
        "One small turtle step at a time..."
      </footer>
    </div>
  );
}
