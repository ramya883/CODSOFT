export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  matchedPattern?: string;
}

export interface ChatPattern {
  patterns: RegExp[];
  responses: string[];
  category?: string;
  keywords?: string[];
}

export interface TrainedPattern {
  id: string;
  question: string;
  answer: string;
  keywords: string[];
  category: string;
  confidenceScore: number;
}

export interface SynonymMap {
  [key: string]: string[];
}

export interface MatchResult {
  response: string;
  confidence: number;
  matchedPattern?: string;
  keywords?: string[];
}
