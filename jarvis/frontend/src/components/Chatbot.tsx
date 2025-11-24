import React, { useState } from 'react';

const Chatbot: React.FC = () => {
  const [messages, setMessages] = useState<string[]>([]);
  const [input, setInput] = useState('');

  const sendMessage = () => {
    if (input.trim()) {
      setMessages([...messages, `You: ${input}`, 'Jarvis: Hello! This is a placeholder response.']);
      setInput('');
    }
  };

  return (
    <div className="chatbot bg-gray-100 p-4 rounded-lg">
      <div className="messages mb-4 h-64 overflow-y-auto">
        {messages.map((msg, index) => (
          <div key={index} className="message mb-2">{msg}</div>
        ))}
      </div>
      <div className="input flex">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          className="flex-1 p-2 border rounded-l"
          placeholder="Ask Jarvis something..."
        />
        <button onClick={sendMessage} className="bg-blue-500 text-white p-2 rounded-r">Send</button>
      </div>
    </div>
  );
};

export default Chatbot;
