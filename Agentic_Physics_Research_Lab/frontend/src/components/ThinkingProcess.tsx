import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Loader2, BrainCircuit } from 'lucide-react';

interface ThinkingStep {
    content: string;
    timestamp?: number;
}

interface ThinkingProcessProps {
    steps: ThinkingStep[];
    isThinking: boolean;
}

export const ThinkingProcess: React.FC<ThinkingProcessProps> = ({ steps, isThinking }) => {
    const [isOpen, setIsOpen] = useState(true);

    if (steps.length === 0 && !isThinking) return null;

    return (
        <div className="mb-4 rounded-lg overflow-hidden border border-tokyonight-comment/30 bg-tokyonight-dark/30 animate-fade-in">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center justify-between p-3 text-xs font-mono text-tokyonight-comment hover:bg-tokyonight-comment/10 transition-colors"
            >
                <div className="flex items-center gap-2">
                    {isThinking ? (
                        <Loader2 className="w-3 h-3 animate-spin text-tokyonight-cyan" />
                    ) : (
                        <BrainCircuit className="w-3 h-3 text-tokyonight-purple" />
                    )}
                    <span>{isThinking ? 'RESEARCHING...' : 'RESEARCHING COMPLETED'}</span>
                </div>
                {isOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>

            {isOpen && (
                <div className="p-3 pt-0 text-sm font-mono space-y-2 border-t border-tokyonight-comment/10">
                    {steps.map((step, idx) => (
                        <div key={idx} className="flex gap-2 animate-slide-up">
                            <span className="text-tokyonight-comment select-none">
                                {String(idx + 1).padStart(2, '0')}
                            </span>
                            <span className="text-tokyonight-cyan/90 break-words">{step.content}</span>
                        </div>
                    ))}
                    {isThinking && (
                        <div className="flex gap-2 animate-pulse">
                            <span className="text-tokyonight-comment select-none">
                                {String(steps.length + 1).padStart(2, '0')}
                            </span>
                            <span className="h-4 w-2 bg-tokyonight-cyan/50 inline-block"></span>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};
