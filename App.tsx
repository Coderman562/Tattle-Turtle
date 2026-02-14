
import React, { useState, useEffect, useRef } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';
import { EscalationType, PatternTracker, TeacherAlert, TurtleResponse, UrgencyLevel, SessionState, InteractionMode } from './types';
import { getTurtleSupport } from './services/geminiService';
import { determineEscalation, detectHelpRequest, detectHighRiskContent, generateTeacherAlert } from './utils/escalationLogic';
import { deleteTeacherAlert, getStudentPatterns, getTeacherAlerts, logInternalConcern, markEscalation, saveTeacherAlert } from './utils/patternTracking';

// --- Audio Utils ---
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

function normalizeSafetyIntentText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s']/g, ' ')
    .replace(/\b(pls|plz|plss)\b/g, 'please')
    .replace(/\b(halp|hep)\b/g, 'help')
    .replace(/\b(2)\b/g, 'to')
    .replace(/\b(u)\b/g, 'you')
    .replace(/\b(techer|tacher|teachr|teaher|teaccher|teecher)\b/g, 'teacher')
    .replace(/\b(dont)\b/g, "don't")
    .replace(/\s+/g, ' ')
    .trim();
}

function hasTeacherHelpIntent(text: string): boolean {
  const normalized = normalizeSafetyIntentText(text);
  const patterns = [
    /\b(i need to (talk|speak) to (a |the |my )?teacher|i want to (talk|speak) to (a |the |my )?teacher)\b/i,
    /\b((talk|speak) to (a |the |my )?teacher)\b/i,
    /\b(i need (a |the |my )?teacher|see (a |the |my )?teacher)\b/i,
    /\b(tell (a |the |my )?teacher|get (a |the |my )?teacher|call (a |the |my )?teacher)\b/i,
    /\b(i need help|please help me|can someone help me)\b/i,
  ];

  return patterns.some((pattern) => pattern.test(normalized));
}

function hasStopOrTeacherRequest(text: string): boolean {
  const normalized = normalizeSafetyIntentText(text);
  const patterns = [
    /\b(i'?m done|im done|that'?s all|can i go|never mind|forget it)\b/i,
    /\b(stop|please stop|stop talking|don'?t want to talk|no more talking|end (this )?chat)\b/i,
    /\b(i need to (talk|speak) to (a |the |my )?teacher|i need (a |the |my )?teacher|i want to (talk|speak) to (a |the |my )?teacher)\b/i,
    /\b((talk|speak) to (a |the |my )?teacher|see (a |the |my )?teacher)\b/i,
    /\b(tell (a |the |my )?teacher|get (a |the |my )?teacher|call (a |the |my )?teacher)\b/i,
    /\b(i need help|please help me|can someone help me|help me now)\b/i,
    /\b(this is serious|i don'?t feel safe|not safe)\b/i,
  ];

  return patterns.some((pattern) => pattern.test(normalized));
}

function shouldForceTeacherEscalation(
  latestStudentText: string,
  conversationHistory: string,
  patterns: PatternTracker[],
): boolean {
  if (
    detectHighRiskContent(latestStudentText) ||
    detectHelpRequest(latestStudentText) ||
    hasTeacherHelpIntent(latestStudentText)
  ) {
    return true;
  }

  return determineEscalation(latestStudentText, conversationHistory, patterns) === EscalationType.IMMEDIATE;
}

export default function App() {
  const [state, setState] = useState<SessionState>({
    step: 'WELCOME',
    inputMode: 'VOICE',
    interactionMode: 'LISTENING',
    transcript: '',
    response: null
  });
  const [error, setError] = useState<string | null>(null);
  const [activeResultCard, setActiveResultCard] = useState<'MISSION' | 'METER' | null>(null);
  const [volume, setVolume] = useState(0);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [needsEscalationConfirmation, setNeedsEscalationConfirmation] = useState(false);
  const [teacherContactStatus, setTeacherContactStatus] = useState<string | null>(null);
  const [showTeacherDashboard, setShowTeacherDashboard] = useState(false);
  const [teacherAlerts, setTeacherAlerts] = useState<TeacherAlert[]>([]);
  const [teacherDashboardToast, setTeacherDashboardToast] = useState<string | null>(null);

  // Audio Context Refs
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const conversationHistoryRef = useRef<string>('');
  const sessionPromiseRef = useRef<any>(null);
  const studentIdRef = useRef<string>('anonymous-student');
  const isEndingSessionRef = useRef(false);
  const forceTeacherEscalationRef = useRef(false);
  const recentStudentTextRef = useRef('');
  const alertSentThisSessionRef = useRef(false);
  const lastSeenTeacherAlertRef = useRef<string | null>(null);
  
  // Silence Detection
  const silenceTimerRef = useRef<any>(null);
  const silencePromptsRef = useRef([
    "It's okay. Take your time.",
    "I'm still here if you want to share.",
    "You can start anywhere you like.",
    "Even a little bit is okay."
  ]);
  const currentSilenceIndexRef = useRef(0);

  const resetSilenceTimer = () => {
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    
    silenceTimerRef.current = setTimeout(() => {
      if (state.step === 'VOICE_CHAT' && sessionPromiseRef.current && !isSpeaking) {
        const prompt = silencePromptsRef.current[currentSilenceIndexRef.current];
        currentSilenceIndexRef.current = (currentSilenceIndexRef.current + 1) % silencePromptsRef.current.length;
        
        sessionPromiseRef.current.then((session: any) => {
          session.send({
            clientContent: {
              turns: [{ role: 'user', parts: [{ text: `[SYSTEM: The child is hesitant. Say exactly: "${prompt}"]` }] }],
              turnComplete: true
            }
          });
        });
      }
    }, 4000);
  };

  useEffect(() => {
    return () => {
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    };
  }, [state.step, isSpeaking]);

  useEffect(() => {
    const key = 'tattle_turtle_student_id_v1';
    const existing = localStorage.getItem(key);
    if (existing) {
      studentIdRef.current = existing;
      return;
    }

    const generated = `student-${Math.random().toString(36).slice(2, 10)}`;
    studentIdRef.current = generated;
    localStorage.setItem(key, generated);
  }, []);

  useEffect(() => {
    setTeacherAlerts(getTeacherAlerts());
  }, []);

  useEffect(() => {
    if (!showTeacherDashboard) {
      return;
    }

    const refresh = () => setTeacherAlerts(getTeacherAlerts());
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

  const stopRealtimeResources = () => {
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    activeSourcesRef.current.forEach(s => s.stop());
    activeSourcesRef.current.clear();
    setIsSpeaking(false);

    scriptProcessorRef.current?.disconnect();
    scriptProcessorRef.current = null;

    mediaStreamRef.current?.getTracks().forEach(track => track.stop());
    mediaStreamRef.current = null;
  };

  const triggerImmediateTeacherAlert = (summary: string) => {
    if (alertSentThisSessionRef.current) {
      return;
    }

    const alert: TeacherAlert = {
      timestamp: new Date().toISOString(),
      escalationType: EscalationType.IMMEDIATE,
      urgency: UrgencyLevel.RED,
      summary: summary.split(/\s+/).slice(0, 20).join(' '),
      concernCategory: 'emotional_regulation',
      primaryEmotion: 'unspecified',
      patternFlag: false,
      studentConfirmedEscalation: true,
      actionSuggestion: 'Check in with student privately',
    };

    saveTeacherAlert(alert);
    markEscalation(studentIdRef.current, EscalationType.IMMEDIATE);
    setTeacherAlerts(getTeacherAlerts());
    setTeacherContactStatus('Teacher has been notified to check in privately.');
    window.alert('New teacher alert created. Please check in with the student now.');
    alertSentThisSessionRef.current = true;
  };

  const startVoiceSession = async (customGreeting?: string) => {
    setError(null);
    isEndingSessionRef.current = false;
    forceTeacherEscalationRef.current = false;
    alertSentThisSessionRef.current = false;
    setTeacherContactStatus(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        } 
      });
      mediaStreamRef.current = stream;
      
      setState(prev => ({ ...prev, step: 'VOICE_CHAT' }));
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      nextStartTimeRef.current = 0;

      const greeting = customGreeting || "Hey friend! Want to tell me something?";
      
      const systemInstruction = `
        - You are Tattle Turtle, a gentle, patient, and warm active listener for kids (6-10).
        - CONVERSATION START: You MUST speak first. Your first words must be: "${greeting}".
        - DIALOGUE RULES:
          - Keep every message UNDER 2 short sentences.
          - Speak at least once every 1-2 turns.
          - If the student is vague (e.g., "something happened"), ask ONE gentle, specific follow-up. Focus on EITHER "what happened" OR "how it felt". Never both at once.
          - Example prompts: "Do you want to tell me what happened?", "How did that make you feel?", "Was that happy, sad, or something else?"
        - SILENCE & HESITATION:
          - If the child is quiet, use normalize stopping if it persists: "That's okay. We can talk again later if you want."
          - Otherwise, follow internal nudge instructions from the system.
        - DO NOT give advice. Only listen, acknowledge, and guide with gentle questions.
        ${state.interactionMode === 'SOCRATIC' ? '- Use Socratic questioning techniques (Categorization, Conceptualization, etc.) sparingly and gently.' : ''}
      `;

      sessionPromiseRef.current = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-12-2025',
        callbacks: {
          onopen: () => {
            // SPEECH FIRST: Force the initial greeting immediately on open
            sessionPromiseRef.current.then((session: any) => {
              session.send({
                clientContent: {
                  turns: [{ role: 'user', parts: [{ text: `[START SESSION: Greet the child immediately with: "${greeting}"]` }] }],
                  turnComplete: true
                }
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
              for(let i=0; i<inputData.length; i++) {
                sum += inputData[i] * inputData[i];
                if (Math.abs(inputData[i]) > peak) peak = Math.abs(inputData[i]);
              }
              const rms = Math.sqrt(sum / inputData.length);
              setVolume(Math.min(100, rms * 500));

              // Only reset silence timer if the user is actually making noise
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
            if (isEndingSessionRef.current) {
              return;
            }

            if (message.serverContent?.inputTranscription) {
              const childText = message.serverContent.inputTranscription.text;
              conversationHistoryRef.current += ' Student: ' + childText;
              recentStudentTextRef.current = `${recentStudentTextRef.current} ${childText}`.slice(-600);
              resetSilenceTimer();

              const patterns = getStudentPatterns(studentIdRef.current);
              const stopCandidate = `${childText} ${recentStudentTextRef.current}`;
              if (shouldForceTeacherEscalation(stopCandidate, conversationHistoryRef.current, patterns)) {
                forceTeacherEscalationRef.current = true;
                triggerImmediateTeacherAlert('Student asked for teacher support during voice chat.');
                endVoiceSession();
                return;
              }

              if (hasStopOrTeacherRequest(stopCandidate)) {
                if (hasTeacherHelpIntent(stopCandidate)) {
                  forceTeacherEscalationRef.current = true;
                  triggerImmediateTeacherAlert('Student requested teacher support.');
                }
                endVoiceSession();
                return;
              }
            }
            
            if (message.serverContent?.outputTranscription) {
              conversationHistoryRef.current += ' Turtle: ' + message.serverContent.outputTranscription.text;
              resetSilenceTimer();
            }

            const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (base64Audio) {
              setIsSpeaking(true);
              const ctx = outputAudioContextRef.current!;
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
              const buffer = await decodeAudioData(decode(base64Audio), ctx, 24000, 1);
              const source = ctx.createBufferSource();
              source.buffer = buffer;
              source.connect(ctx.destination);
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += buffer.duration;
              activeSourcesRef.current.add(source);
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
            console.error('Turtle API Error:', e);
            setError("Turtle is having a little trouble hearing. Let's try again!");
            resetSession();
          },
          onclose: (e: CloseEvent) => {
            console.debug('Session closed');
          },
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } },
          systemInstruction,
          inputAudioTranscription: {},
          outputAudioTranscription: {}
        }
      });
    } catch (err: any) {
      setError("Turtle's ears are covered! Please check your microphone permissions.");
      setState(prev => ({ ...prev, step: 'WELCOME' }));
    }
  };

  const endVoiceSession = async () => {
    if (isEndingSessionRef.current) {
      return;
    }
    isEndingSessionRef.current = true;
    stopRealtimeResources();
    setState(prev => ({ ...prev, step: 'PROCESSING' }));
    await sessionPromiseRef.current?.then((session: any) => session.close()).catch(() => {});
    sessionPromiseRef.current = null;
    handleFinalSubmit(conversationHistoryRef.current || "(Silence)");
  };

  const handleFinalSubmit = async (text: string) => {
    setState(prev => ({ ...prev, step: 'PROCESSING' }));
    setTeacherContactStatus(null);
    try {
      const patterns = getStudentPatterns(studentIdRef.current);
      const response = await getTurtleSupport(text, conversationHistoryRef.current || text, patterns, studentIdRef.current);
      const transcriptRequestsTeacher = hasTeacherHelpIntent(text) || hasTeacherHelpIntent(conversationHistoryRef.current || '');
      const shouldAutoNotifyTeacher =
        forceTeacherEscalationRef.current ||
        response.escalationType === EscalationType.IMMEDIATE ||
        transcriptRequestsTeacher;

      if (shouldAutoNotifyTeacher) {
        if (!alertSentThisSessionRef.current) {
          const alert = generateTeacherAlert(response, true);
          saveTeacherAlert(alert);
          markEscalation(studentIdRef.current, response.escalationType || EscalationType.IMMEDIATE);
          setTeacherAlerts(getTeacherAlerts());
          window.alert('New teacher alert created. Please check in with the student now.');
          alertSentThisSessionRef.current = true;
        }
        response.needsEscalationConfirmation = false;
        setTeacherContactStatus('Teacher has been notified to check in privately.');
      }

      setNeedsEscalationConfirmation(Boolean(response.needsEscalationConfirmation) && !shouldAutoNotifyTeacher);
      setState(prev => ({ ...prev, step: 'RESULTS', response }));
    } catch (err: any) {
      setError("Please talk to your teacher! Turtle cannot no longer assist you.");
      setState(prev => ({ ...prev, step: 'WELCOME' }));
    }
  };

  const resetSession = () => {
    isEndingSessionRef.current = false;
    forceTeacherEscalationRef.current = false;
    alertSentThisSessionRef.current = false;
    recentStudentTextRef.current = '';
    stopRealtimeResources();
    sessionPromiseRef.current?.then((session: any) => session.close()).catch(() => {});
    sessionPromiseRef.current = null;
    setState({ step: 'WELCOME', inputMode: 'VOICE', interactionMode: 'LISTENING', transcript: '', response: null });
    setError(null);
    setActiveResultCard(null);
    setNeedsEscalationConfirmation(false);
    setTeacherContactStatus(null);
    conversationHistoryRef.current = '';
  };

  const resumeConversation = () => {
    const followUp = state.response?.followUpQuestion;
    setState(prev => ({ ...prev, step: 'VOICE_CHAT', response: null }));
    startVoiceSession(followUp);
  };

  const confirmEscalation = (confirmed: boolean) => {
    if (!state.response) {
      setNeedsEscalationConfirmation(false);
      return;
    }

    if (confirmed) {
      const alert = generateTeacherAlert(state.response, true);
      saveTeacherAlert(alert);
      markEscalation(studentIdRef.current, state.response.escalationType || EscalationType.IMMEDIATE);
      setTeacherAlerts(getTeacherAlerts());
      window.alert('New teacher alert created. Please check in with the student now.');
      setTeacherContactStatus('Teacher has been notified to check in privately.');
    } else {
      logInternalConcern(
        studentIdRef.current,
        state.response.summary || 'Escalation declined by student.',
        state.response.escalationType || EscalationType.IMMEDIATE,
      );
      setTeacherContactStatus('Okay. We will not notify a teacher right now.');
    }

    setNeedsEscalationConfirmation(false);
  };

  return (
    <div className="min-h-screen px-4 py-10 md:px-12 md:py-16 flex flex-col items-center">
      {showTeacherDashboard && teacherDashboardToast && (
        <div className="fixed top-6 right-6 z-50 bg-[var(--soft-coral)] text-white px-6 py-4 rounded-[1.25rem] shadow-2xl border-2 border-white">
          <p className="text-xl font-bubble">{teacherDashboardToast}</p>
        </div>
      )}
      <header className="flex flex-col items-center mb-12 text-center">
        <div className="flex items-center gap-6 mb-2 cursor-pointer" onClick={resetSession}>
          <div className="text-8xl turtle-bounce">🐢</div>
          <h1 className="text-6xl font-bubble text-[var(--text-cocoa)] tracking-wide drop-shadow-sm">Tattle Turtle</h1>
        </div>
        <div className="flex gap-3 mb-4">
          <button
            onClick={() => setShowTeacherDashboard(false)}
            className={`bubbly-button px-6 py-3 text-xl font-bubble ${!showTeacherDashboard ? 'bg-[var(--sky-blue)] text-white' : 'bg-white text-[var(--text-cocoa)]'}`}
          >
            Student View
          </button>
          <button
            onClick={() => setShowTeacherDashboard(true)}
            className={`bubbly-button px-6 py-3 text-xl font-bubble ${showTeacherDashboard ? 'bg-[var(--soft-coral)] text-white' : 'bg-white text-[var(--text-cocoa)]'}`}
          >
            Teacher Dashboard
          </button>
        </div>
        <p className="text-2xl text-[var(--text-clay)] font-bubble italic">Slow and steady, we share our worries.</p>
      </header>

      <main className="w-full max-w-4xl flex flex-col items-center">
        {showTeacherDashboard && (
          <div className="w-full max-w-4xl bubbly-card bg-white p-8 md:p-10">
            <h2 className="text-4xl font-bubble text-[var(--text-cocoa)] mb-6">Teacher Alerts</h2>
            {teacherAlerts.length === 0 ? (
              <p className="text-2xl text-[var(--text-clay)]">No alerts yet.</p>
            ) : (
              <div className="flex flex-col gap-4">
                {teacherAlerts.map((alert, index) => (
                  <div key={`${alert.timestamp}-${index}`} className="rounded-[1.5rem] p-5 border-2 border-[var(--mint-calm)]">
                    <div className="flex justify-between items-start gap-4">
                      <div className="flex flex-wrap gap-3 text-lg text-[var(--text-cocoa)] mb-2">
                        <span className="font-bold">{new Date(alert.timestamp).toLocaleString()}</span>
                        <span className={`px-3 py-1 rounded-full text-white ${
                          alert.urgency === UrgencyLevel.RED
                            ? 'bg-[var(--blush-rose)]'
                            : alert.urgency === UrgencyLevel.YELLOW
                              ? 'bg-[var(--warm-butter)]'
                              : 'bg-[var(--gentle-leaf)]'
                        }`}>
                          {alert.urgency}
                        </span>
                        <span className="px-3 py-1 rounded-full bg-[var(--mint-calm)]">{alert.concernCategory}</span>
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
                    <p className="text-base text-[var(--text-clay)]">
                      Consent: {alert.studentConfirmedEscalation ? 'Yes' : 'No'} | Pattern: {alert.patternFlag ? 'Yes' : 'No'}
                    </p>
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
                onClick={() => setState(s => ({ ...s, interactionMode: 'LISTENING' }))}
                className={`flex-1 py-5 px-6 rounded-[3rem] font-bubble text-2xl transition-all duration-300 ${state.interactionMode === 'LISTENING' ? 'bg-[var(--sky-blue)] text-white shadow-md' : 'text-[var(--text-cocoa)] hover:bg-white/30'}`}
              >
                👂 Ears On
              </button>
              <button
                onClick={() => setState(s => ({ ...s, interactionMode: 'SOCRATIC' }))}
                className={`flex-1 py-5 px-6 rounded-[3rem] font-bubble text-2xl transition-all duration-300 ${state.interactionMode === 'SOCRATIC' ? 'bg-[var(--lavender-wonder)] text-white shadow-md' : 'text-[var(--text-cocoa)] hover:bg-white/30'}`}
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
            
            {error && <p className="text-[var(--soft-coral)] font-bold bg-white px-10 py-4 rounded-full shadow-md text-xl">{error}</p>}
          </div>
        )}

        {state.step === 'VOICE_CHAT' && (
          <div className="flex flex-col items-center gap-10 animate-in slide-in-from-bottom-8 duration-500">
            <div className="relative group">
              <div 
                className={`absolute inset-0 rounded-full blur-3xl opacity-40 transition-all duration-300 ${isSpeaking ? 'bg-[var(--warm-sunshine)] scale-125' : 'bg-[var(--turtle-green)] scale-100'}`}
                style={{ transform: `scale(${1 + (volume / 100)})` }}
              />
              <div className={`w-64 h-64 rounded-full relative z-10 ${state.interactionMode === 'LISTENING' ? 'bg-[var(--mint-calm)]' : 'bg-[var(--lavender-wonder)]'} flex items-center justify-center ${!isSpeaking ? 'listening-pulse' : ''} shadow-2xl transition-all duration-500 ${isSpeaking ? 'scale-110' : ''}`}>
                <span className={`text-[12rem] transition-all duration-300 ${isSpeaking ? 'rotate-3' : 'rotate-0'}`}>🐢</span>
              </div>
            </div>
            
            <div className="text-center">
              <p className="text-4xl font-bubble text-[var(--text-cocoa)] mb-2">
                {isSpeaking ? "Turtle is talking..." : "Turtle is Listening..."}
              </p>
              <div className="flex gap-1 justify-center h-8 items-center">
                {[1,2,3,4,5].map(i => (
                  <div 
                    key={i} 
                    className={`w-2 rounded-full transition-all duration-75 ${isSpeaking ? 'bg-[var(--warm-sunshine)]' : 'bg-[var(--sky-blue)]'}`}
                    style={{ 
                      opacity: (volume > (i * 15) || isSpeaking) ? 0.8 : 0.2, 
                      height: (volume > (i * 15) || (isSpeaking && Math.random() > 0.5)) ? '2.5rem' : '0.75rem' 
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
            <button
              onClick={() => {
                forceTeacherEscalationRef.current = true;
                triggerImmediateTeacherAlert('Student pressed Need Teacher Now.');
                endVoiceSession();
              }}
              className="bubbly-button bg-[var(--soft-coral)] hover:opacity-90 text-white text-4xl font-bubble py-7 px-16 shadow-lg transform transition-hover active:scale-95"
            >
              Need Teacher Now
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
                {state.response.sufficient 
                  ? (state.interactionMode === 'LISTENING' ? "I heard every bit! ❤️" : "We did some great thinking! 🧠")
                  : ""
                }
              </p>
              {state.response.sufficient && (
                <p className="text-2xl text-[var(--text-clay)] font-medium">Choose a card to see what we found:</p>
              )}
            </div>

            {state.response.shouldEndConversation ? (
              <div className="bg-[var(--gentle-leaf)] bubbly-card p-10 text-center w-full max-w-2xl">
                <p className="text-4xl font-bubble text-white">
                  {state.response.closingMessage || "I'm glad we talked. Take care!"}
                </p>
              </div>
            ) : state.response.sufficient ? (
              <>
                <div className="flex flex-col sm:flex-row gap-8 w-full max-w-3xl">
                  <button
                    onClick={() => setActiveResultCard(activeResultCard === 'MISSION' ? null : 'MISSION')}
                    className={`flex-1 bubbly-button text-3xl font-bubble py-10 px-8 flex flex-col items-center gap-4 transition-all duration-300 ${activeResultCard === 'MISSION' ? 'bg-[var(--warm-sunshine)] scale-105 shadow-xl' : 'bg-[var(--warm-sunshine)] opacity-70 hover:opacity-100'} text-[var(--text-cocoa)]`}
                  >
                    <span>Tiny Brave Mission 🏅</span>
                  </button>
                  <button
                    onClick={() => setActiveResultCard(activeResultCard === 'METER' ? null : 'METER')}
                    className={`flex-1 bubbly-button text-3xl font-bubble py-10 px-8 flex flex-col items-center gap-4 transition-all duration-300 ${activeResultCard === 'METER' ? 'bg-[var(--soft-coral)] scale-105 shadow-xl' : 'bg-[var(--soft-coral)] opacity-70 hover:opacity-100'} text-white`}
                  >
                    <span>Big Feelings Meter 🎨</span>
                  </button>
                </div>

                <div className="w-full mt-6 min-h-[350px] flex justify-center transition-all duration-500">
                  {activeResultCard === 'MISSION' && (
                    <div className="bubbly-card w-full max-w-2xl bg-[var(--mint-calm)] p-12 animate-in slide-in-from-top-6 duration-500 text-center shadow-inner">
                      <h3 className="text-5xl font-bubble text-[var(--text-cocoa)] mb-8">Your Mission 🏅</h3>
                      <p className="text-3xl text-[var(--text-clay)] font-bubble mb-10 italic">"{state.response.exercise.task}"</p>
                      <div className="inline-block bg-[var(--bubblegum-pink)] px-12 py-5 rounded-full border-4 border-white shadow-md transform rotate-2">
                        <span className="text-3xl font-bubble text-white uppercase tracking-widest animate-pulse">
                          ✨ {state.response.exercise.reward} ✨
                        </span>
                      </div>
                    </div>
                  )}

                  {activeResultCard === 'METER' && (
                    <div className="bubbly-card w-full max-w-2xl bg-[var(--mint-calm)] p-12 animate-in slide-in-from-top-6 duration-500 text-center shadow-inner">
                      <h3 className="text-5xl font-bubble text-[var(--text-cocoa)] mb-8">Feelings Meter 🎨</h3>
                      <div className="mb-10 flex justify-center">
                        <UrgencyBadge level={state.response.urgency} />
                      </div>
                      <p className="text-3xl text-[var(--text-clay)] font-bubble italic leading-relaxed">
                        "{state.response.summary}"
                      </p>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="flex flex-col items-center gap-8 py-12 animate-in slide-in-from-bottom-6">
                <p className="text-3xl font-bubble text-[var(--text-clay)] max-w-md text-center italic">
                   "{state.response.listeningHealing}"
                </p>
              </div>
            )}

            {needsEscalationConfirmation && state.response.escalationType === EscalationType.IMMEDIATE && (
              <div className="bg-[var(--warm-sunshine)] bubbly-card p-12 mt-4 w-full max-w-2xl animate-in zoom-in duration-500">
                <p className="text-3xl font-bubble text-[var(--text-cocoa)] mb-6 text-center">
                  This sounds really important. I think your teacher should know so they can help. Is that okay with you?
                </p>
                <div className="flex gap-6">
                  <button
                    onClick={() => confirmEscalation(true)}
                    className="flex-1 bubbly-button bg-[var(--gentle-leaf)] text-white text-3xl py-6"
                  >
                    Yes, tell my teacher
                  </button>
                  <button
                    onClick={() => confirmEscalation(false)}
                    className="flex-1 bubbly-button bg-[var(--text-clay)] text-white text-3xl py-6"
                  >
                    Not right now
                  </button>
                </div>
              </div>
            )}

            {teacherContactStatus && (
              <div className="bg-white bubbly-card p-6 w-full max-w-2xl border-4 border-[var(--gentle-leaf)]">
                <p className="text-2xl font-bubble text-[var(--text-cocoa)] text-center">{teacherContactStatus}</p>
              </div>
            )}

            {state.response.urgency === UrgencyLevel.RED && !needsEscalationConfirmation && (
              <div className="bg-[var(--blush-rose)] bubbly-card p-12 mt-12 w-full max-w-2xl animate-in zoom-in duration-500">
                <h3 className="text-5xl font-bubble text-white mb-6 text-center">Wait! 🆘</h3>
                <p className="text-2xl text-white mb-8 font-medium text-center leading-relaxed">
                  This sounds like a very big worry. It's best to talk to a teacher you trust right away.
                </p>
                <div className="bg-white/90 p-10 rounded-[2.5rem] mb-10 text-2xl text-[var(--text-cocoa)] italic font-bubble text-center">
                  "{state.response.helpInstruction}"
                </div>
                <div className="flex gap-6">
                  <button
                    onClick={() => {
                      if (state.response) {
                        const alert = generateTeacherAlert(state.response, true);
                        saveTeacherAlert(alert);
                        markEscalation(studentIdRef.current, state.response.escalationType || EscalationType.IMMEDIATE);
                        setTeacherAlerts(getTeacherAlerts());
                        window.alert('New teacher alert created. Please check in with the student now.');
                        setTeacherContactStatus('Teacher has been notified to check in privately.');
                      }
                    }}
                    className="flex-1 bubbly-button bg-white text-[var(--soft-coral)] text-3xl font-bubble py-6 shadow-md hover:bg-white/100"
                  >
                    I will talk to a teacher
                  </button>
                  <button
                    onClick={() => {
                      if (state.response) {
                        logInternalConcern(
                          studentIdRef.current,
                          state.response.summary || 'Student deferred teacher outreach.',
                          state.response.escalationType || EscalationType.IMMEDIATE,
                        );
                        setTeacherContactStatus('Okay. We will not notify a teacher right now.');
                      }
                    }}
                    className="flex-1 bubbly-button bg-[var(--soft-coral)] border-4 border-white text-white text-3xl font-bubble py-6"
                  >
                    Not now
                  </button>
                </div>
              </div>
            )}

            <button
              onClick={resetSession}
              className="mt-12 bubbly-button bg-[var(--mint-calm)] text-[var(--text-cocoa)] text-3xl font-bubble py-5 px-16 transition-all hover:opacity-80"
            >
              Start New Story
            </button>
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

const UrgencyBadge: React.FC<{ level: UrgencyLevel }> = ({ level }) => {
  const config = {
    [UrgencyLevel.GREEN]: { color: 'bg-[var(--gentle-leaf)]', label: 'Cool 🍀' },
    [UrgencyLevel.YELLOW]: { color: 'bg-[var(--warm-butter)]', label: 'Steady ⚖️' },
    [UrgencyLevel.RED]: { color: 'bg-[var(--blush-rose)]', label: 'Big 🦸' },
  };
  const { color, label } = config[level] || config[UrgencyLevel.GREEN];
  return (
    <span className={`px-14 py-5 rounded-full text-4xl font-bubble tracking-wider uppercase shadow-md text-white ${color}`}>
      {label}
    </span>
  );
}
