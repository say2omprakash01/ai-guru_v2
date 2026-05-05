/** Mirrors `api/main.py` response models (JSON field names). */

export type CitationOut = {
  scripture_name: string;
  chapter: string;
  verse: string;
  similarity: number;
  internal_id: number;
};

export type RetrievedVerseOut = {
  rank: number;
  citation: string;
  chapter: string;
  verse: string;
  shloka_text: string;
  meaning: string;
  similarity: number;
};

export type EmotionPredictionOut = {
  label: string;
  confidence: number;
  scores: Record<string, number>;
};

export type GuidanceBodyOut = {
  emotion: string;
  insight: string;
  explanation: string;
  practical_guidance: string;
  reflection_question: string;
  disclaimer: string;
};

export type GuidanceResponseOut = {
  ok: boolean;
  message: string;
  top_k: number;
  citations: CitationOut[];
  retrieved_verses: RetrievedVerseOut[];
  emotion_prediction: EmotionPredictionOut | null;
  guidance: GuidanceBodyOut;
  llm_model: string | null;
};
