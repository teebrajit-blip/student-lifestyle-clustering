import type {
  PatientInfo,
  SymptomInput,
  AnalysisResult,
  Hospital,
  RankedHospital,
  TriageLevel,
} from './types';
import { HOSPITALS } from '../data/demo';

const RED_FLAG_SYMPTOMS = new Set([
  'chest_pain', 'shortness_breath', 'fainting', 'seizure', 'bleeding',
  'injury', 'burns', 'confusion', 'stroke_face', 'stroke_speech',
  'allergic_reaction', 'pregnancy_issue', 'child_emergency',
]);

const SPECIALTY_MAP: Record<string, { primary: string; secondary: string; requirement: string }> = {
  chest_pain: { primary: 'Emergency Medicine', secondary: 'Cardiology', requirement: 'cardiology' },
  shortness_breath: { primary: 'Emergency Medicine', secondary: 'Pulmonology', requirement: 'icu' },
  palpitations: { primary: 'Cardiology', secondary: 'Emergency Medicine', requirement: 'cardiology' },
  sweating: { primary: 'Cardiology', secondary: 'Emergency Medicine', requirement: 'cardiology' },
  headache: { primary: 'Neurology', secondary: 'Emergency Medicine', requirement: 'neurology' },
  dizziness: { primary: 'Neurology', secondary: 'Emergency Medicine', requirement: 'neurology' },
  fainting: { primary: 'Emergency Medicine', secondary: 'Neurology', requirement: 'icu' },
  seizure: { primary: 'Emergency Medicine', secondary: 'Neurology', requirement: 'neurology' },
  stroke_face: { primary: 'Emergency Medicine', secondary: 'Neurology', requirement: 'neurology' },
  stroke_speech: { primary: 'Emergency Medicine', secondary: 'Neurology', requirement: 'neurology' },
  confusion: { primary: 'Emergency Medicine', secondary: 'Neurology', requirement: 'neurology' },
  abdominal_pain: { primary: 'General Medicine', secondary: 'Surgery', requirement: 'surgery' },
  vomiting: { primary: 'General Medicine', secondary: 'Gastroenterology', requirement: 'icu' },
  diarrhea: { primary: 'General Medicine', secondary: 'Gastroenterology', requirement: 'icu' },
  fever: { primary: 'General Medicine', secondary: 'Internal Medicine', requirement: 'icu' },
  weakness: { primary: 'General Medicine', secondary: 'Internal Medicine', requirement: 'icu' },
  cough: { primary: 'Pulmonology', secondary: 'General Medicine', requirement: 'icu' },
  bleeding: { primary: 'Emergency Medicine', secondary: 'Trauma Surgery', requirement: 'trauma' },
  injury: { primary: 'Emergency Medicine', secondary: 'Orthopedics', requirement: 'trauma' },
  burns: { primary: 'Emergency Medicine', secondary: 'Burn Unit', requirement: 'burnUnit' },
  allergic_reaction: { primary: 'Emergency Medicine', secondary: 'Immunology', requirement: 'icu' },
  pregnancy_issue: { primary: 'Obstetrics & Gynecology', secondary: 'Emergency Medicine', requirement: 'obstetrics' },
  child_emergency: { primary: 'Pediatrics', secondary: 'Emergency Medicine', requirement: 'pediatrics' },
  rash: { primary: 'Dermatology', secondary: 'General Medicine', requirement: 'icu' },
  urinary_issue: { primary: 'General Medicine', secondary: 'Nephrology', requirement: 'dialysis' },
};

const CONDITION_MAP: Record<string, { name: string; description: string }[]> = {
  chest_pain: [
    { name: 'Acute cardiac event (e.g. myocardial infarction)', description: 'Sudden chest pain with breathing difficulty may indicate a cardiac emergency.' },
    { name: 'Severe respiratory condition', description: 'Acute breathing distress requiring urgent evaluation.' },
    { name: 'Musculoskeletal chest pain', description: 'Non-cardiac chest wall pain, less urgent but needs review.' },
  ],
  shortness_breath: [
    { name: 'Acute respiratory distress', description: 'Sudden inability to breathe comfortably may need oxygen support.' },
    { name: 'Asthma / COPD exacerbation', description: 'Worsening of chronic airway disease.' },
    { name: 'Cardiac-related breathlessness', description: 'Heart-related fluid backup in lungs.' },
  ],
  headache: [
    { name: 'Migraine', description: 'Severe recurrent headache, often one-sided.' },
    { name: 'Tension headache', description: 'Stress-related tightness around the head.' },
    { name: 'Possible neurological cause', description: 'Sudden severe headache needs ruling out serious causes.' },
  ],
  fever: [
    { name: 'Viral fever / flu-like illness', description: 'Common self-limiting infection.' },
    { name: 'Dengue-like illness', description: 'Fever with body ache; platelet monitoring advised.' },
    { name: 'Bacterial infection', description: 'May need lab work and physician review.' },
  ],
  abdominal_pain: [
    { name: 'Acute gastroenteritis', description: 'Inflammation of stomach and intestines.' },
    { name: 'Appendicitis (if right-sided)', description: 'Surgical cause requiring urgent evaluation.' },
    { name: 'Gallbladder / liver issue', description: 'Upper abdominal pain after meals.' },
  ],
  seizure: [
    { name: 'Epileptic seizure', description: 'Sudden uncontrolled electrical brain activity.' },
    { name: 'Seizure from metabolic cause', description: 'Low sugar or electrolyte imbalance.' },
    { name: 'First-time seizure of unknown cause', description: 'Needs urgent neurological evaluation.' },
  ],
  stroke_face: [
    { name: 'Stroke (cerebrovascular event)', description: 'Face drooping or one-sided weakness is a stroke warning.' },
    { name: 'Transient ischemic attack', description: 'Temporary stroke-like symptoms needing urgent review.' },
  ],
};

function pickConditions(symptoms: string[]): { name: string; description: string }[] {
  for (const s of symptoms) {
    if (CONDITION_MAP[s]) return CONDITION_MAP[s];
  }
  return [
    { name: 'Non-specific illness', description: 'Symptoms need clinical evaluation to identify the cause.' },
    { name: 'Possible infection', description: 'Generalized symptoms may indicate an infection.' },
    { name: 'Other undetermined condition', description: 'Further assessment by a healthcare professional recommended.' },
  ];
}

function determineTriage(
  symptoms: string[],
  severity: number,
  age: number,
  onset: string,
): TriageLevel {
  const hasRedFlag = symptoms.some((s) => RED_FLAG_SYMPTOMS.has(s));
  if (hasRedFlag && severity >= 6) return 'critical';
  if (hasRedFlag) return 'urgent';
  if (severity >= 8) return 'urgent';
  if (severity >= 5 || (age > 60 && severity >= 4)) return 'moderate';
  return 'low';
}

function computeRiskScore(
  symptoms: string[],
  severity: number,
  age: number,
  onset: string,
  history: string[],
): number {
  let score = 0;
  const redCount = symptoms.filter((s) => RED_FLAG_SYMPTOMS.has(s)).length;
  score += redCount * 22;
  score += Math.round(severity * 4);
  if (age > 60) score += 12;
  else if (age > 45) score += 6;
  if (onset === 'sudden') score += 8;
  if (history.includes('Heart disease') || history.includes('Hypertension')) score += 8;
  if (history.includes('Diabetes')) score += 5;
  return Math.min(100, Math.max(8, score));
}

function buildRiskFactors(
  symptoms: string[],
  severity: number,
  age: number,
  onset: string,
  history: string[],
) {
  const factors = [
    { label: 'Red-flag symptom present', present: symptoms.some((s) => RED_FLAG_SYMPTOMS.has(s)) },
    { label: `Severity ${severity}/10`, present: severity >= 6 },
    { label: 'Sudden onset', present: onset === 'sudden' },
    { label: 'Age > 60', present: age > 60 },
    { label: 'Age > 45', present: age > 45 },
    { label: 'Cardiac history', present: history.some((h) => ['Heart disease', 'Hypertension'].includes(h)) },
    { label: 'Diabetes', present: history.includes('Diabetes') },
    { label: 'Respiratory history', present: history.includes('Asthma') || history.includes('COPD') },
  ];
  return factors.filter((f) => f.present);
}

function buildRedFlags(symptoms: string[]): string[] {
  const labels: Record<string, string> = {
    chest_pain: 'Severe chest pain',
    shortness_breath: 'Severe difficulty breathing',
    fainting: 'Unconsciousness / fainting',
    seizure: 'Active seizure',
    bleeding: 'Severe bleeding',
    injury: 'Major trauma',
    burns: 'Severe burns',
    stroke_face: 'Stroke-like symptoms (face drooping)',
    stroke_speech: 'Slurred speech',
    allergic_reaction: 'Severe allergic reaction',
    pregnancy_issue: 'Pregnancy emergency',
    child_emergency: 'Child emergency',
  };
  return symptoms.filter((s) => labels[s]).map((s) => labels[s]);
}

function buildDoctorSummary(
  patient: PatientInfo,
  input: SymptomInput,
  result: Omit<AnalysisResult, 'doctorSummary'>,
): string {
  const lines: string[] = [];
  lines.push('PATIENT EMERGENCY SUMMARY');
  lines.push('');
  lines.push(`Age: ${patient.age || 'Unknown'}`);
  lines.push(`Gender: ${patient.gender || 'Unknown'}`);
  lines.push(`Blood Group: ${patient.bloodGroup || 'Unknown'}`);
  lines.push('');
  lines.push('Main Symptoms:');
  input.selectedSymptoms.forEach((sid) => {
    const found = sid;
    lines.push(`• ${found.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}`);
  });
  if (input.text.trim()) lines.push(`• "${input.text.trim()}"`);
  lines.push('');
  lines.push(`Duration: ${input.duration || 'Not specified'}`);
  lines.push(`Onset: ${input.onset || 'Not specified'}`);
  lines.push(`Severity: ${input.severity}/10`);
  lines.push('');
  if (patient.existingConditions.length) {
    lines.push(`Relevant History: ${patient.existingConditions.join(', ')}`);
  }
  if (patient.allergies.length) lines.push(`Allergies: ${patient.allergies.join(', ')}`);
  if (patient.currentMedications.trim()) lines.push(`Current Medications (patient-reported): ${patient.currentMedications}`);
  lines.push('');
  lines.push(`AI Triage: ${result.triage.toUpperCase()}`);
  lines.push(`AI Risk Indicator: ${result.riskScore}/100`);
  lines.push('');
  lines.push(`Suggested Department: ${result.recommendedSpecialty}`);
  if (result.additionalSpecialty) lines.push(`Additional: ${result.additionalSpecialty}`);
  lines.push('');
  lines.push('Possible Conditions:');
  result.possibleConditions.forEach((c, i) => {
    lines.push(`${i + 1}. ${c.name} (${c.confidence}%)`);
  });
  lines.push('');
  lines.push('Important: AI-generated support summary. Not a confirmed diagnosis.');
  lines.push('Recommended Action: Immediate professional medical evaluation.');
  return lines.join('\n');
}

export function analyzeSymptoms(
  patient: PatientInfo,
  input: SymptomInput,
): AnalysisResult {
  const age = parseInt(patient.age || '0', 10) || 0;
  const symptoms = input.selectedSymptoms;
  const triage = determineTriage(symptoms, input.severity, age, input.onset);
  const riskScore = computeRiskScore(symptoms, input.severity, age, input.onset, patient.existingConditions);
  const riskFactors = buildRiskFactors(symptoms, input.severity, age, input.onset, patient.existingConditions);
  const conditions = pickConditions(symptoms).map((c, i) => ({
    name: c.name,
    description: c.description,
    confidence: [78, 14, 8][i] ?? 5,
  }));
  const specialtyKey = symptoms.find((s) => SPECIALTY_MAP[s]) || '';
  const specialty = SPECIALTY_MAP[specialtyKey] || {
    primary: 'General Medicine',
    secondary: 'Emergency Medicine',
    requirement: 'icu',
  };
  const redFlags = buildRedFlags(symptoms);
  const hospitalRequirements = [specialty.requirement, 'emergency'];

  const partial = {
    triage,
    riskScore,
    riskFactors,
    possibleConditions: conditions,
    recommendedSpecialty: specialty.primary,
    additionalSpecialty: specialty.secondary,
    hospitalRequirements,
    redFlags,
  };

  const doctorSummary = buildDoctorSummary(patient, input, partial);

  return { ...partial, doctorSummary };
}

const WEIGHTS = {
  specialtyMatch: 0.35,
  distance: 0.2,
  emergencyCapability: 0.15,
  availability: 0.1,
  specialist: 0.1,
  travelTime: 0.05,
  rating: 0.05,
};

export function rankHospitals(
  hospitals: Hospital[],
  requirements: string[],
  triage: TriageLevel,
): RankedHospital[] {
  const ranked = hospitals.map((h) => {
    const specialtyMatch = requirements.some((r) => h.facilities[r as keyof Hospital['facilities']])
      ? 1
      : 0;
    const distanceScore = Math.max(0, 1 - h.distanceKm / 10);
    const emergencyCap = h.facilities.emergency ? 1 : 0.4;
    const availability = h.icuBedsTotal > 0 ? h.icuBedsAvailable / h.icuBedsTotal : 0.5;
    const specialist = h.specialistAvailable ? 1 : 0.5;
    const travelScore = Math.max(0, 1 - h.etaMin / 30);
    const ratingScore = h.rating / 5;

    const score =
      specialtyMatch * WEIGHTS.specialtyMatch * 100 +
      distanceScore * WEIGHTS.distance * 100 +
      emergencyCap * WEIGHTS.emergencyCapability * 100 +
      availability * WEIGHTS.availability * 100 +
      specialist * WEIGHTS.specialist * 100 +
      travelScore * WEIGHTS.travelTime * 100 +
      ratingScore * WEIGHTS.rating * 100;

    const breakdown = [
      { label: 'Specialty Match', weight: WEIGHTS.specialtyMatch, value: Math.round(specialtyMatch * WEIGHTS.specialtyMatch * 100) },
      { label: 'Distance', weight: WEIGHTS.distance, value: Math.round(distanceScore * WEIGHTS.distance * 100) },
      { label: 'Emergency Capability', weight: WEIGHTS.emergencyCapability, value: Math.round(emergencyCap * WEIGHTS.emergencyCapability * 100) },
      { label: 'Availability', weight: WEIGHTS.availability, value: Math.round(availability * WEIGHTS.availability * 100) },
      { label: 'Specialist On-site', weight: WEIGHTS.specialist, value: Math.round(specialist * WEIGHTS.specialist * 100) },
      { label: 'Travel Time', weight: WEIGHTS.travelTime, value: Math.round(travelScore * WEIGHTS.travelTime * 100) },
      { label: 'Rating', weight: WEIGHTS.rating, value: Math.round(ratingScore * WEIGHTS.rating * 100) },
    ];

    return { ...h, score: Math.round(score), scoreBreakdown: breakdown, bestMatch: false };
  });

  ranked.sort((a, b) => b.score - a.score);
  if (ranked.length > 0) ranked[0].bestMatch = true;
  // Triage boost: critical cases weight emergency + ICU more
  if (triage === 'critical') {
    ranked.sort((a, b) => {
      const aw = (a.facilities.emergency ? 1 : 0) + (a.facilities.icu ? 1 : 0) + a.icuBedsAvailable * 0.1;
      const bw = (b.facilities.emergency ? 1 : 0) + (b.facilities.icu ? 1 : 0) + b.icuBedsAvailable * 0.1;
      return b.score + bw * 5 - (a.score + aw * 5);
    });
    if (ranked.length > 0) {
      ranked.forEach((r) => (r.bestMatch = false));
      ranked[0].bestMatch = true;
    }
  }
  return ranked;
}

export function getDemoHospitals(): Hospital[] {
  return HOSPITALS;
}
