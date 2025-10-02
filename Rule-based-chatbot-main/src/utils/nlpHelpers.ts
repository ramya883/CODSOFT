import { SynonymMap } from '../types/chatbot';

const stopWords = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
  'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
  'to', 'was', 'will', 'with', 'you', 'your', 'my', 'i', 'me'
]);

export const synonyms: SynonymMap = {
  'hello': ['hi', 'hey', 'greetings', 'howdy', 'hola', 'welcome'],
  'goodbye': ['bye', 'farewell', 'see you', 'later', 'cya', 'leaving'],
  'help': ['assist', 'support', 'aid', 'guide', 'show'],
  'thanks': ['thank you', 'appreciate', 'grateful', 'thx', 'ty'],
  'good': ['great', 'excellent', 'awesome', 'nice', 'wonderful', 'amazing'],
  'bad': ['terrible', 'awful', 'poor', 'horrible', 'worst'],
  'yes': ['yeah', 'yep', 'sure', 'ok', 'okay', 'affirmative', 'correct'],
  'no': ['nope', 'nah', 'negative', 'not really', 'never'],
  'how': ['what', 'which', 'when', 'where', 'why'],
  'tell': ['show', 'explain', 'describe', 'say', 'inform'],
  'know': ['understand', 'learn', 'aware', 'familiar', 'recognize'],
  'make': ['create', 'build', 'construct', 'develop', 'produce'],
  'get': ['obtain', 'receive', 'acquire', 'fetch', 'retrieve'],
  'want': ['need', 'desire', 'wish', 'like', 'prefer'],
  'think': ['believe', 'consider', 'feel', 'suppose', 'assume'],
  'find': ['search', 'locate', 'discover', 'seek', 'look for'],
  'work': ['function', 'operate', 'run', 'perform', 'execute'],
  'use': ['utilize', 'employ', 'apply', 'implement', 'operate'],
  'give': ['provide', 'offer', 'supply', 'deliver', 'present'],
  'love': ['like', 'enjoy', 'adore', 'appreciate', 'prefer'],
  'hate': ['dislike', 'detest', 'despise', 'loathe'],
  'big': ['large', 'huge', 'massive', 'enormous', 'gigantic'],
  'small': ['tiny', 'little', 'mini', 'compact', 'petite'],
  'fast': ['quick', 'rapid', 'swift', 'speedy', 'hasty'],
  'slow': ['sluggish', 'gradual', 'leisurely', 'unhurried'],
  'happy': ['joyful', 'pleased', 'cheerful', 'delighted', 'content'],
  'sad': ['unhappy', 'depressed', 'miserable', 'sorrowful', 'gloomy'],
  'smart': ['intelligent', 'clever', 'bright', 'brilliant', 'wise'],
  'stupid': ['dumb', 'foolish', 'silly', 'ignorant', 'unintelligent']
};

export function extractKeywords(text: string): string[] {
  const words = text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(word => word.length > 2 && !stopWords.has(word));

  return [...new Set(words)];
}

export function expandWithSynonyms(word: string): string[] {
  const normalized = word.toLowerCase();
  const expanded = [normalized];

  if (synonyms[normalized]) {
    expanded.push(...synonyms[normalized]);
  }

  for (const [key, syns] of Object.entries(synonyms)) {
    if (syns.includes(normalized)) {
      expanded.push(key, ...syns);
    }
  }

  return [...new Set(expanded)];
}

export function calculateSimilarity(keywords1: string[], keywords2: string[]): number {
  if (keywords1.length === 0 || keywords2.length === 0) return 0;

  const expanded1 = new Set(keywords1.flatMap(expandWithSynonyms));
  const expanded2 = new Set(keywords2.flatMap(expandWithSynonyms));

  const intersection = new Set([...expanded1].filter(x => expanded2.has(x)));
  const union = new Set([...expanded1, ...expanded2]);

  return intersection.size / union.size;
}

export function fuzzyMatch(input: string, target: string, threshold: number = 0.7): boolean {
  const inputWords = extractKeywords(input);
  const targetWords = extractKeywords(target);

  return calculateSimilarity(inputWords, targetWords) >= threshold;
}

export function extractIntent(text: string): string {
  const lowerText = text.toLowerCase();

  const intentPatterns: { [key: string]: RegExp[] } = {
    'question': [/^(what|where|when|who|why|how|can|could|would|should|is|are|do|does)/i],
    'command': [/^(tell|show|give|find|get|make|create|help)/i],
    'statement': [/^(i |my |the |this |that )/i],
    'greeting': [/^(hi|hello|hey|greetings)/i],
    'farewell': [/^(bye|goodbye|see you|farewell)/i]
  };

  for (const [intent, patterns] of Object.entries(intentPatterns)) {
    for (const pattern of patterns) {
      if (pattern.test(lowerText)) {
        return intent;
      }
    }
  }

  return 'unknown';
}
