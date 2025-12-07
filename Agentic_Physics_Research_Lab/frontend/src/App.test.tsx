import { render, screen, fireEvent } from '@testing-library/react';
import App from './App';
import { describe, it, expect, vi } from 'vitest';

describe('App', () => {
    it('renders the chat interface', () => {
        render(<App />);
        expect(screen.getByText('Physics Copilot')).toBeInTheDocument();
    });

    it('allows entering text', () => {
        render(<App />);
        const input = screen.getByPlaceholderText('Ask a physics question...');
        fireEvent.change(input, { target: { value: 'What is the speed of light?' } });
        expect(input).toHaveValue('What is the speed of light?');
    });
});
