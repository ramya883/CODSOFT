import { useState, useEffect, useRef } from 'react';
import { MessageCircle, Brain, Lightbulb } from 'lucide-react';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import TrainingPanel from './components/TrainingPanel';
import { Message } from './types/chatbot';
import { getBotResponse } from './utils/chatbotRules';

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "Hello! I'm an intelligent rule-based chatbot with natural language understanding. I can discuss technology, science, history, entertainment, and many other topics. I also understand synonyms and related words! Try asking me something, or click the brain icon to train me with custom responses.",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const [showTraining, setShowTraining] = useState(false);
  const [conversationContext, setConversationContext] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = (text: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      text,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setConversationContext(prev => [...prev.slice(-5), text]);
    setIsTyping(true);

    setTimeout(() => {
      const result = getBotResponse(text, conversationContext);
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: result.response,
        sender: 'bot',
        timestamp: new Date(),
        matchedPattern: result.matchedPattern
      };

      setMessages(prev => [...prev, botMessage]);
      setConversationContext(prev => [...prev.slice(-5), result.response]);
      setIsTyping(false);
    }, 500 + Math.random() * 1000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 flex items-center justify-center p-4">
      <div className="w-full max-w-4xl h-[90vh] bg-white rounded-3xl shadow-2xl flex flex-col overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 to-cyan-600 px-6 py-5 flex items-center gap-3 shadow-md">
          <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center">
            <MessageCircle className="text-blue-600" size={24} />
          </div>
          <div className="flex-1">
            <h1 className="text-xl font-bold text-white">Smart Chatbot</h1>
            <p className="text-blue-100 text-sm">Natural language understanding with training</p>
          </div>
          <button
            onClick={() => setShowTraining(true)}
            className="flex items-center gap-2 bg-white bg-opacity-20 hover:bg-opacity-30 text-white px-4 py-2 rounded-full transition-all"
            title="Training Mode"
          >
            <Brain size={20} />
            <span className="text-sm font-medium">Train</span>
          </button>
          <Lightbulb className="text-blue-200" size={24} />
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4 bg-gray-50">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}

          {isTyping && (
            <div className="flex gap-3 justify-start">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center">
                <MessageCircle size={18} className="text-white" />
              </div>
              <div className="bg-white rounded-2xl px-4 py-3 shadow-sm">
                <div className="flex gap-1">
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className="px-6 py-4 bg-white border-t border-gray-200">
          <ChatInput onSendMessage={handleSendMessage} disabled={isTyping} />
        </div>
      </div>

      {showTraining && (
        <TrainingPanel
          onClose={() => setShowTraining(false)}
          onPatternAdded={() => {
            const successMessage: Message = {
              id: Date.now().toString(),
              text: "Great! I've learned something new. Try asking me about it!",
              sender: 'bot',
              timestamp: new Date()
            };
            setMessages(prev => [...prev, successMessage]);
          }}
        />
      )}
    </div>
  );
}

export default App;
