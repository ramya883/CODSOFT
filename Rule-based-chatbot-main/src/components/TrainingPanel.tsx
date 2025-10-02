import { useState } from 'react';
import { Brain, Plus, Trash2, X } from 'lucide-react';
import { TrainedPattern } from '../types/chatbot';
import { addTrainedPattern, deleteTrainedPattern, getTrainedPatterns } from '../utils/chatbotRules';

interface TrainingPanelProps {
  onClose: () => void;
  onPatternAdded: () => void;
}

export default function TrainingPanel({ onClose, onPatternAdded }: TrainingPanelProps) {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [category, setCategory] = useState('custom');
  const [patterns, setPatterns] = useState<TrainedPattern[]>(getTrainedPatterns());

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (question.trim() && answer.trim()) {
      addTrainedPattern(question, answer, category);
      setQuestion('');
      setAnswer('');
      setPatterns(getTrainedPatterns());
      onPatternAdded();
    }
  };

  const handleDelete = (id: string) => {
    deleteTrainedPattern(id);
    setPatterns(getTrainedPatterns());
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-3xl shadow-2xl w-full max-w-3xl max-h-[85vh] overflow-hidden flex flex-col">
        <div className="bg-gradient-to-r from-green-500 to-emerald-600 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Brain className="text-white" size={24} />
            <h2 className="text-xl font-bold text-white">Training Mode</h2>
          </div>
          <button
            onClick={onClose}
            className="text-white hover:bg-white hover:bg-opacity-20 rounded-full p-2 transition-all"
          >
            <X size={20} />
          </button>
        </div>

        <div className="p-6 space-y-6 overflow-y-auto flex-1">
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
            <p className="text-sm text-blue-800">
              Teach the chatbot new responses! Enter a question or phrase and the response you want the bot to give.
              The bot will learn to recognize similar questions using keyword matching.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Question or Phrase
              </label>
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="e.g., What is your favorite color?"
                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Response
              </label>
              <textarea
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                placeholder="e.g., I don't have personal preferences, but blue is often considered calming!"
                rows={3}
                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent resize-none"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Category
              </label>
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
              >
                <option value="custom">Custom</option>
                <option value="personal">Personal</option>
                <option value="knowledge">Knowledge</option>
                <option value="preference">Preference</option>
                <option value="opinion">Opinion</option>
              </select>
            </div>

            <button
              type="submit"
              className="w-full bg-gradient-to-r from-green-500 to-emerald-600 text-white py-3 rounded-xl font-semibold hover:from-green-600 hover:to-emerald-700 transition-all flex items-center justify-center gap-2 shadow-md"
            >
              <Plus size={20} />
              Add Training Pattern
            </button>
          </form>

          {patterns.length > 0 && (
            <div className="space-y-3">
              <h3 className="font-semibold text-gray-800 flex items-center gap-2">
                <Brain size={18} />
                Trained Patterns ({patterns.length})
              </h3>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {patterns.map((pattern) => (
                  <div
                    key={pattern.id}
                    className="bg-gray-50 rounded-xl p-4 border border-gray-200 hover:border-green-300 transition-all"
                  >
                    <div className="flex justify-between items-start gap-3">
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-gray-800 mb-1 break-words">
                          Q: {pattern.question}
                        </p>
                        <p className="text-sm text-gray-600 mb-2 break-words">
                          A: {pattern.answer}
                        </p>
                        <div className="flex gap-2 flex-wrap">
                          <span className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded-full">
                            {pattern.category}
                          </span>
                          <span className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded-full">
                            Used {pattern.confidenceScore} times
                          </span>
                          <div className="flex gap-1">
                            {pattern.keywords.slice(0, 3).map((keyword, idx) => (
                              <span
                                key={idx}
                                className="text-xs px-2 py-1 bg-gray-200 text-gray-700 rounded-full"
                              >
                                {keyword}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={() => handleDelete(pattern.id)}
                        className="flex-shrink-0 text-red-500 hover:bg-red-50 p-2 rounded-lg transition-all"
                      >
                        <Trash2 size={18} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
