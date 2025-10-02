import { ChatPattern, TrainedPattern, MatchResult } from '../types/chatbot';
import { extractKeywords, calculateSimilarity, extractIntent } from './nlpHelpers';

export const chatRules: ChatPattern[] = [
  {
    patterns: [/^(hi|hello|hey|greetings|howdy)/i],
    responses: [
      "Hello! How can I help you today?",
      "Hi there! What can I do for you?",
      "Hey! Nice to see you. What's on your mind?"
    ],
    category: 'greeting',
    keywords: ['hello', 'hi', 'greetings']
  },
  {
    patterns: [/^(bye|goodbye|see you|farewell|later)/i],
    responses: [
      "Goodbye! Have a great day!",
      "See you later! Feel free to come back anytime.",
      "Bye! It was nice chatting with you."
    ],
    category: 'farewell',
    keywords: ['goodbye', 'bye', 'farewell']
  },
  {
    patterns: [/how are you|how do you do|what's up|how's it going/i],
    responses: [
      "I'm just a chatbot, but I'm functioning perfectly! How are you?",
      "I'm doing great, thanks for asking! How can I assist you?",
      "All systems running smoothly! What brings you here today?"
    ],
    category: 'smalltalk',
    keywords: ['how', 'you', 'doing']
  },
  {
    patterns: [/what is your name|who are you|your name|introduce yourself/i],
    responses: [
      "I'm a smart rule-based chatbot here to help answer your questions!",
      "You can call me ChatBot. I'm here to assist you with various queries.",
      "I'm an AI assistant powered by advanced pattern matching and keyword analysis."
    ],
    category: 'identity',
    keywords: ['name', 'who', 'identity']
  },
  {
    patterns: [/help|assist|support|guide me/i],
    responses: [
      "I can help you with many topics! Try asking about technology, science, history, or just have a casual conversation. You can also train me with new responses!",
      "I'm here to assist! Ask me questions, request information, or teach me new things using the training mode.",
      "Need help? I understand natural language and can discuss various topics. What would you like to know?"
    ],
    category: 'help',
    keywords: ['help', 'assist', 'support']
  },
  {
    patterns: [/weather|temperature|forecast|rain|sunny|climate/i],
    responses: [
      "I don't have real-time weather data, but I can discuss climate patterns and weather phenomena if you'd like!",
      "While I can't check current weather, I can tell you about meteorology concepts. What interests you?",
      "Weather questions need real-time data I don't have access to, but I'd be happy to discuss weather science!"
    ],
    category: 'weather',
    keywords: ['weather', 'temperature', 'forecast']
  },
  {
    patterns: [/joke|funny|laugh|humor|make me laugh/i],
    responses: [
      "Why don't scientists trust atoms? Because they make up everything!",
      "What do you call a bear with no teeth? A gummy bear!",
      "Why did the scarecrow win an award? He was outstanding in his field!",
      "How does a computer get drunk? It takes screenshots!",
      "Why do programmers prefer dark mode? Because light attracts bugs!"
    ],
    category: 'joke',
    keywords: ['joke', 'funny', 'humor']
  },
  {
    patterns: [/thank you|thanks|appreciate|grateful/i],
    responses: [
      "You're welcome! Happy to help!",
      "No problem at all! Let me know if you need anything else.",
      "Glad I could help! Feel free to ask more questions."
    ],
    category: 'gratitude',
    keywords: ['thanks', 'appreciate', 'grateful']
  },
  {
    patterns: [/what can you do|your capabilities|what do you know|abilities/i],
    responses: [
      "I can understand natural language, recognize keywords and synonyms, answer questions on various topics, tell jokes, and learn from training! I use advanced pattern matching with keyword analysis.",
      "My capabilities include: understanding context, matching similar phrases, discussing multiple topics, and being trained with new responses. What would you like to explore?",
      "I'm trained on various topics and can recognize related words and phrases. Ask me about technology, science, general knowledge, or teach me something new!"
    ],
    category: 'capabilities',
    keywords: ['capabilities', 'abilities', 'can', 'do']
  },
  {
    patterns: [/time|what time|clock|hour/i],
    responses: [
      `The current time is ${new Date().toLocaleTimeString()}.`,
      `It's ${new Date().toLocaleTimeString()} right now.`,
      `According to my system, it's ${new Date().toLocaleTimeString()}.`
    ],
    category: 'time',
    keywords: ['time', 'clock', 'hour']
  },
  {
    patterns: [/date|today|what day|calendar/i],
    responses: [
      `Today is ${new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}.`,
      `The date is ${new Date().toLocaleDateString()}.`,
      `It's ${new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}.`
    ],
    category: 'date',
    keywords: ['date', 'today', 'day']
  },
  {
    patterns: [/computer|technology|software|programming|code/i],
    responses: [
      "Technology is fascinating! Computers work by processing binary data through electronic circuits. What aspect interests you?",
      "I love discussing technology! From hardware to software, programming languages to algorithms - what would you like to know?",
      "Computing technology has revolutionized our world. Are you curious about programming, AI, hardware, or something else?"
    ],
    category: 'technology',
    keywords: ['computer', 'technology', 'programming']
  },
  {
    patterns: [/science|physics|chemistry|biology|experiment/i],
    responses: [
      "Science is the systematic study of the natural world through observation and experimentation. Which field interests you most?",
      "I can discuss various scientific topics! Physics, chemistry, biology, astronomy - what would you like to explore?",
      "Science helps us understand how the universe works. What scientific concept would you like to learn about?"
    ],
    category: 'science',
    keywords: ['science', 'physics', 'chemistry', 'biology']
  },
  {
    patterns: [/history|past|ancient|historical|civilization/i],
    responses: [
      "History teaches us about human civilization and how we got here. What historical period or event interests you?",
      "From ancient civilizations to modern times, history is full of fascinating stories. What would you like to discuss?",
      "Understanding history helps us learn from the past. Which era or topic would you like to explore?"
    ],
    category: 'history',
    keywords: ['history', 'past', 'ancient']
  },
  {
    patterns: [/music|song|artist|melody|instrument/i],
    responses: [
      "Music is a universal language that transcends cultures! What genre or aspect of music interests you?",
      "From classical to contemporary, music has incredible diversity. Do you have favorite artists or instruments?",
      "Music can evoke powerful emotions and memories. What kind of music do you enjoy?"
    ],
    category: 'music',
    keywords: ['music', 'song', 'artist']
  },
  {
    patterns: [/book|read|literature|story|novel|author/i],
    responses: [
      "Books open doors to countless worlds and ideas! What genres do you enjoy reading?",
      "Literature reflects human experience across cultures and times. Do you have favorite authors or books?",
      "Reading is one of humanity's greatest inventions. What type of stories captivate you?"
    ],
    category: 'literature',
    keywords: ['book', 'literature', 'reading']
  },
  {
    patterns: [/food|eat|cooking|recipe|meal|restaurant/i],
    responses: [
      "Food brings people together and reflects cultural diversity! What cuisines do you enjoy?",
      "Cooking is both art and science. Do you like to cook, or prefer dining out?",
      "From simple meals to gourmet cuisine, food is essential to human culture. What's your favorite dish?"
    ],
    category: 'food',
    keywords: ['food', 'cooking', 'meal']
  },
  {
    patterns: [/sport|game|exercise|fitness|training|athlete/i],
    responses: [
      "Sports combine physical skill with strategy and teamwork. What sports do you follow or play?",
      "Fitness and exercise are important for health and wellbeing. What's your preferred way to stay active?",
      "From individual sports to team competitions, athletics showcase human achievement. What interests you?"
    ],
    category: 'sports',
    keywords: ['sport', 'game', 'exercise']
  },
  {
    patterns: [/movie|film|cinema|director|actor/i],
    responses: [
      "Cinema is a powerful storytelling medium! What genres or directors do you appreciate?",
      "Films can entertain, educate, and inspire. Do you have favorite movies or actors?",
      "From blockbusters to indie films, cinema offers incredible variety. What do you like to watch?"
    ],
    category: 'entertainment',
    keywords: ['movie', 'film', 'cinema']
  },
  {
    patterns: [/travel|vacation|trip|destination|country|city/i],
    responses: [
      "Travel broadens perspectives and creates lasting memories! Where have you been or want to go?",
      "Exploring new places is one of life's great adventures. What destinations interest you?",
      "Different countries and cities offer unique cultures and experiences. What's on your travel wishlist?"
    ],
    category: 'travel',
    keywords: ['travel', 'vacation', 'destination']
  }
];

export let trainedPatterns: TrainedPattern[] = [];

function initializeTrainingData() {
  const trainingData = [
    { q: "Hi", a: "Hello! How are you today?", c: "greeting" },
    { q: "Hello", a: "Hello! How are you today?", c: "greeting" },
    { q: "Hey", a: "Hello! How are you today?", c: "greeting" },
    { q: "Good morning", a: "Hello! How are you today?", c: "greeting" },
    { q: "How are you?", a: "I'm doing great, thanks for asking! How about you?", c: "greeting" },
    { q: "How's it going?", a: "I'm doing great, thanks for asking! How about you?", c: "greeting" },
    { q: "What's your name?", a: "I'm your intelligent chatbot, here to chat with you about technology, science, history, and more!", c: "greeting" },
    { q: "Who are you?", a: "I'm your intelligent chatbot, here to chat with you about technology, science, history, and more!", c: "greeting" },
    { q: "Nice to meet you", a: "Nice to meet you too!", c: "greeting" },
    { q: "Glad to see you", a: "Nice to meet you too!", c: "greeting" },
    { q: "Bye", a: "Goodbye! Take care and come back soon.", c: "greeting" },
    { q: "Goodbye", a: "Goodbye! Take care and come back soon.", c: "greeting" },
    { q: "See you later", a: "Goodbye! Take care and come back soon.", c: "greeting" },
    { q: "Thank you", a: "You're welcome! Always happy to help.", c: "greeting" },
    { q: "Thanks a lot", a: "You're welcome! Always happy to help.", c: "greeting" },
    { q: "Tell me about yourself", a: "I'm a rule-based chatbot with natural language understanding. I can discuss multiple domains like tech, science, history, and entertainment.", c: "greeting" },
    { q: "Can you remember my name is Ramya?", a: "Yes! I'll remember that your name is Ramya.", c: "personal" },
    { q: "What's my name?", a: "You told me your name is Ramya.", c: "personal" },
    { q: "Do you like me?", a: "Of course! I enjoy chatting with you.", c: "personal" },

    { q: "What is AI?", a: "Artificial Intelligence is the simulation of human intelligence in machines.", c: "technology" },
    { q: "Explain artificial intelligence", a: "Artificial Intelligence is the simulation of human intelligence in machines.", c: "technology" },
    { q: "What is machine learning?", a: "Machine Learning is a branch of AI where systems learn patterns from data without being explicitly programmed.", c: "technology" },
    { q: "What is blockchain?", a: "Blockchain is a decentralized ledger of transactions. Cryptocurrency, like Bitcoin, uses blockchain to operate securely.", c: "technology" },
    { q: "What is cryptocurrency?", a: "Blockchain is a decentralized ledger of transactions. Cryptocurrency, like Bitcoin, uses blockchain to operate securely.", c: "technology" },
    { q: "Who founded Microsoft?", a: "Microsoft was founded by Bill Gates and Paul Allen in 1975.", c: "technology" },
    { q: "Who is Elon Musk?", a: "Elon Musk is the founder of Tesla, SpaceX, and Neuralink.", c: "technology" },
    { q: "What is the Internet of Things (IoT)?", a: "IoT is a network of physical devices connected to the internet, sharing data automatically.", c: "technology" },
    { q: "What is cloud computing?", a: "Cloud computing means storing and accessing data and applications over the internet instead of on your computer.", c: "technology" },
    { q: "What is 5G?", a: "5G is the fifth generation of mobile networks, offering faster speeds and low latency.", c: "technology" },
    { q: "What is virtual reality?", a: "Virtual reality is a simulated 3D environment that immerses users through VR headsets.", c: "technology" },
    { q: "What is cybersecurity?", a: "Cybersecurity is the practice of protecting systems, networks, and data from digital attacks.", c: "technology" },

    { q: "What is photosynthesis?", a: "Photosynthesis is the process where plants use sunlight, carbon dioxide, and water to make food.", c: "science" },
    { q: "What is gravity?", a: "Gravity is the force that attracts objects toward each other, discovered by Isaac Newton.", c: "science" },
    { q: "What is the speed of light?", a: "The speed of light is about 299,792 km per second in a vacuum.", c: "science" },
    { q: "Who is Albert Einstein?", a: "Albert Einstein was a physicist best known for the theory of relativity.", c: "science" },
    { q: "What is DNA?", a: "DNA is the molecule that carries genetic information in living organisms.", c: "science" },
    { q: "What is evolution?", a: "Evolution is the process by which species change over time through natural selection.", c: "science" },
    { q: "What is an atom?", a: "An atom is the basic unit of matter, made up of protons, neutrons, and electrons.", c: "science" },
    { q: "What is the human brain?", a: "The human brain is the central organ of the nervous system, controlling thoughts, memory, and body functions.", c: "science" },
    { q: "What is the Milky Way?", a: "The Milky Way is the galaxy that contains our solar system.", c: "science" },
    { q: "What is a black hole?", a: "A black hole is a region in space where gravity is so strong that not even light can escape.", c: "science" },

    { q: "Who was the first President of the USA?", a: "George Washington was the first President of the United States.", c: "history" },
    { q: "Who was Mahatma Gandhi?", a: "Mahatma Gandhi was the leader of India's independence movement using non-violence.", c: "history" },
    { q: "When did World War II end?", a: "World War II ended in 1945.", c: "history" },
    { q: "Who was Napoleon Bonaparte?", a: "Napoleon was a French military leader who became Emperor of France.", c: "history" },
    { q: "Who built the pyramids?", a: "The ancient Egyptians built the pyramids as tombs for pharaohs.", c: "history" },
    { q: "Who discovered America?", a: "Christopher Columbus is often credited with discovering America in 1492.", c: "history" },
    { q: "What is the Industrial Revolution?", a: "The Industrial Revolution was a period of rapid industrial growth during the 18th and 19th centuries.", c: "history" },
    { q: "Who was Nelson Mandela?", a: "Nelson Mandela was a South African leader who fought against apartheid and became President.", c: "history" },
    { q: "When did India gain independence?", a: "India gained independence from British rule on August 15, 1947.", c: "history" },
    { q: "Who was Julius Caesar?", a: "Julius Caesar was a Roman general and statesman who played a key role in the fall of the Roman Republic.", c: "history" },

    { q: "Recommend me a movie", a: "You might enjoy Inception if you like sci-fi, or The Shawshank Redemption for a classic.", c: "entertainment" },
    { q: "Who is Iron Man in Marvel?", a: "Iron Man is played by Robert Downey Jr. in the Marvel movies.", c: "entertainment" },
    { q: "Who is Harry Potter?", a: "Harry Potter is a fictional wizard created by author J.K. Rowling.", c: "entertainment" },
    { q: "Who is Mickey Mouse?", a: "Mickey Mouse is a famous Disney cartoon character created in 1928.", c: "entertainment" },
    { q: "Tell me a joke", a: "Why don't programmers like nature? Too many bugs!", c: "entertainment" },
    { q: "Tell me a riddle", a: "Sure! What has keys but can't open locks? (Answer: A piano.)", c: "entertainment" },
    { q: "Who is the King of Pop?", a: "Michael Jackson is known as the King of Pop.", c: "entertainment" },
    { q: "Suggest me a song", a: "You might like \"Shape of You\" by Ed Sheeran.", c: "entertainment" },
    { q: "Who is Sherlock Holmes?", a: "Sherlock Holmes is a fictional detective created by Sir Arthur Conan Doyle.", c: "entertainment" },
    { q: "Tell me a fun fact", a: "Did you know honey never spoils? Archaeologists found pots of honey thousands of years old still edible!", c: "entertainment" }
  ];

  trainingData.forEach((item, index) => {
    const keywords = extractKeywords(item.q);
    trainedPatterns.push({
      id: `trained-${index}`,
      question: item.q,
      answer: item.a,
      keywords,
      category: item.c,
      confidenceScore: 0
    });
  });
}

initializeTrainingData();

export let trainedPatternsInitialized = true;

export function addTrainedPattern(question: string, answer: string, category: string = 'custom'): TrainedPattern {
  const keywords = extractKeywords(question);
  const newPattern: TrainedPattern = {
    id: Date.now().toString(),
    question: question.trim(),
    answer: answer.trim(),
    keywords,
    category,
    confidenceScore: 0
  };

  trainedPatterns.push(newPattern);
  return newPattern;
}

export function getTrainedPatterns(): TrainedPattern[] {
  return trainedPatterns;
}

export function deleteTrainedPattern(id: string): boolean {
  const index = trainedPatterns.findIndex(p => p.id === id);
  if (index !== -1) {
    trainedPatterns.splice(index, 1);
    return true;
  }
  return false;
}

export function getBotResponse(userInput: string, conversationContext: string[] = []): MatchResult {
  const userKeywords = extractKeywords(userInput);
  const intent = extractIntent(userInput);

  let bestMatch: MatchResult = {
    response: '',
    confidence: 0
  };

  for (const trained of trainedPatterns) {
    const similarity = calculateSimilarity(userKeywords, trained.keywords);

    if (similarity > bestMatch.confidence && similarity > 0.3) {
      bestMatch = {
        response: trained.answer,
        confidence: similarity,
        matchedPattern: trained.question,
        keywords: trained.keywords
      };

      trained.confidenceScore++;
    }
  }

  if (bestMatch.confidence > 0.5) {
    return bestMatch;
  }

  for (const rule of chatRules) {
    for (const pattern of rule.patterns) {
      if (pattern.test(userInput)) {
        const responses = rule.responses;
        const selectedResponse = responses[Math.floor(Math.random() * responses.length)];
        return {
          response: selectedResponse,
          confidence: 1.0,
          matchedPattern: rule.category,
          keywords: rule.keywords
        };
      }
    }

    if (rule.keywords) {
      const similarity = calculateSimilarity(userKeywords, rule.keywords);
      if (similarity > bestMatch.confidence && similarity > 0.4) {
        const responses = rule.responses;
        const selectedResponse = responses[Math.floor(Math.random() * responses.length)];
        bestMatch = {
          response: selectedResponse,
          confidence: similarity,
          matchedPattern: rule.category,
          keywords: rule.keywords
        };
      }
    }
  }

  if (bestMatch.confidence > 0.3) {
    return bestMatch;
  }

  const contextualResponses = [
    "That's an interesting question! While I don't have a specific answer, I'm learning. You can teach me by using the training mode!",
    "I'm not entirely sure about that. Could you rephrase it, or would you like to train me with the correct response?",
    "I don't have information on that topic yet. You can help me learn by adding it to my training data!",
    `Based on the keywords I detected: ${userKeywords.join(', ')}, I'm not sure how to respond. Can you teach me?`
  ];

  return {
    response: contextualResponses[Math.floor(Math.random() * contextualResponses.length)],
    confidence: 0,
    keywords: userKeywords
  };
}
