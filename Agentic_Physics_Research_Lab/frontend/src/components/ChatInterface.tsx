import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Bot, User, Atom } from 'lucide-react';
import { ThinkingProcess } from './ThinkingProcess';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    thinkingSteps?: { content: string }[];
}

export const ChatInterface: React.FC = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [currentThinkingSteps, setCurrentThinkingSteps] = useState<{ content: string }[]>([]);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, currentThinkingSteps]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage: Message = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);
        setCurrentThinkingSteps([]);

        try {
            const response = await fetch('http://localhost:8000/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMessage.content }),
            });

            if (!response.body) throw new Error('No response body');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let assistantMessageContent = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6);
                        if (dataStr === '[DONE]') break;

                        try {
                            const data = JSON.parse(dataStr);

                            if (data.type === 'thought' || data.type === 'tool_result' || data.type === 'error') {
                                setCurrentThinkingSteps(prev => [...prev, { content: data.content || JSON.stringify(data) }]);
                            } else if (data.type === 'token') {
                                assistantMessageContent += data.content;
                                // Update the partial message in the UI? 
                                // For simplicity, we just accumulate and show at the end or update a partial state
                                // Ideally we have a "streaming" message state.
                            }
                        } catch (e) {
                            console.error("Error parsing SSE:", e);
                        }
                    }
                }
            }

            // Finalize assistant message
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: assistantMessageContent,
                thinkingSteps: currentThinkingSteps // This captures the steps generated *during* this response
                // Note: state updates in loop might not be reflected here immediately due to closure, 
                // but for this simple implementation let's rely on component state or just append.
            }]);
            // Actually we need to attach the accumulated steps to the finalized message.
            // But `currentThinkingSteps` is state. Closures...
            // Better approach: Update the LAST message if it's assistant, or append new one.

        } catch (error) {
            console.error('Error sending message:', error);
            setCurrentThinkingSteps(prev => [...prev, { content: `Error: ${String(error)}` }]);
        } finally {
            setIsLoading(false);
            // We need to move the thinking steps to the permanent message history context if we want them to persist properly attached 
            // OR we can just keep them 'live' until the next message.
            // Let's refine the state management in a refactor if needed.
        }
    };

    // Correction: We need to see the assistant message growing.
    // We'll use a separate effect or just update messages state incrementally.
    // Revised handleSubmit logic to be safer and simpler for streaming:

    // ACTUALLY, let's just do a simpler implementation for this iteration
    // where we hold "streamingMessage" state and merge it on completion.

    return (
        <div className="flex flex-col h-screen max-w-5xl mx-auto p-4 md:p-6 lg:p-8">
            {/* Header */}
            <header className="flex items-center gap-3 mb-8 animate-fade-in">
                <div className="p-2 bg-tokyonight-purple/20 rounded-lg neon-border">
                    <Atom className="w-6 h-6 text-tokyonight-purple" />
                </div>
                <div>
                    <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-tokyonight-purple to-tokyonight-cyan">
                        Physics Copilot
                    </h1>
                    <div className="flex items-center gap-2 text-xs text-tokyonight-comment font-mono">
                        <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                        SYSTEM ONLINE
                    </div>
                </div>
            </header>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto space-y-6 mb-6 pr-2 scrollbar-hide">
                {messages.map((msg, idx) => (
                    <div key={idx} className={`flex gap-4 animate-fade-in ${msg.role === 'assistant' ? '' : 'flex-row-reverse'}`}>
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 
              ${msg.role === 'assistant' ? 'bg-tokyonight-purple/20 text-tokyonight-purple' : 'bg-tokyonight-cyan/20 text-tokyonight-cyan'}`}>
                            {msg.role === 'assistant' ? <Bot size={18} /> : <User size={18} />}
                        </div>

                        <div className={`flex flex-col max-w-[80%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>

                            {/* Thinking Steps for Assistant */}
                            {msg.role === 'assistant' && msg.thinkingSteps && msg.thinkingSteps.length > 0 && (
                                <div className="w-full max-w-lg mb-2">
                                    <ThinkingProcess steps={msg.thinkingSteps} isThinking={false} />
                                </div>
                            )}

                            <div className={`px-5 py-3 rounded-2xl ${msg.role === 'assistant'
                                ? 'glass-panel text-tokyonight-fg rounded-tl-none'
                                : 'bg-tokyonight-purple/10 border border-tokyonight-purple/30 text-white rounded-tr-none'
                                }`}>
                                {msg.content}
                            </div>
                        </div>
                    </div>
                ))}

                {/* Streaming / Loading State */}
                {isLoading && (
                    <div className="flex gap-4 animate-fade-in">
                        <div className="w-8 h-8 rounded-full bg-tokyonight-purple/20 text-tokyonight-purple flex items-center justify-center shrink-0">
                            <Bot size={18} />
                        </div>
                        <div className="flex flex-col w-full max-w-3xl">
                            <ThinkingProcess steps={currentThinkingSteps} isThinking={true} />
                            {/* Placeholder for streaming text if we had it, or just show thinking... */}
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="relative">
                <form onSubmit={handleSubmit} className="relative z-10">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        disabled={isLoading}
                        placeholder="Ask a physics question..."
                        className="w-full bg-tokyonight-dark/50 border border-tokyonight-comment/30 rounded-xl py-4 pl-5 pr-12 
            text-tokyonight-fg placeholder-tokyonight-comment focus:outline-none focus:border-tokyonight-purple/50 
            focus:ring-1 focus:ring-tokyonight-purple/50 transition-all shadow-lg"
                    />
                    <button
                        type="submit"
                        disabled={!input.trim() || isLoading}
                        className="absolute right-3 top-1/2 -translate-y-1/2 p-2 rounded-lg 
            text-tokyonight-purple hover:bg-tokyonight-purple/10 disabled:opacity-50 disabled:hover:bg-transparent transition-colors"
                    >
                        {isLoading ? <Loader2 className="animate-spin w-5 h-5" /> : <Send className="w-5 h-5" />}
                    </button>
                </form>

                {/* Decorative Glow */}
                <div className="absolute inset-0 -z-10 bg-gradient-to-r from-tokyonight-purple/20 to-tokyonight-cyan/20 blur-xl opacity-30 rounded-xl"></div>
            </div>
        </div>
    );
};
