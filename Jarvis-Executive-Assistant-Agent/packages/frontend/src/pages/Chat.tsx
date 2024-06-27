import { useState, useRef, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Paper,
  TextField,
  IconButton,
  Typography,
  CircularProgress,
  List,
  ListItem,
  Divider,
} from '@mui/material';
import {
  Send as SendIcon,
  AttachFile as AttachFileIcon,
  Stop as StopIcon,
} from '@mui/icons-material';
import { io, Socket } from 'socket.io-client';
import ReactMarkdown from 'react-markdown';
import { useAuth } from '@/hooks/useAuth';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface AgentEvent {
  type: 'thinking' | 'executing_skill' | 'completed' | 'error';
  payload: {
    conversationId: string;
    messageId: string;
    data: any;
  };
  timestamp: Date;
}

const Chat = () => {
  const { conversationId } = useParams();
  const { user } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [socket, setSocket] = useState<Socket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Connect to WebSocket
  useEffect(() => {
    const newSocket = io(import.meta.env.VITE_WS_URL, {
      auth: {
        token: localStorage.getItem('token'),
      },
    });

    newSocket.on('connect', () => {
      console.log('Connected to WebSocket');
    });

    newSocket.on('agent_event', (event: AgentEvent) => {
      handleAgentEvent(event);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from WebSocket');
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleAgentEvent = (event: AgentEvent) => {
    switch (event.type) {
      case 'thinking':
        setIsLoading(true);
        break;
      case 'executing_skill':
        // Could show what skill is being executed
        break;
      case 'completed':
        setIsLoading(false);
        setMessages(prev => [
          ...prev,
          {
            id: event.payload.messageId,
            role: 'assistant',
            content: event.payload.data.message,
            timestamp: new Date(event.timestamp),
          },
        ]);
        break;
      case 'error':
        setIsLoading(false);
        // Handle error - could show an error message
        break;
    }
  };

  const handleSendMessage = () => {
    if (!input.trim() || !socket) return;

    const message: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, message]);
    socket.emit('message', { content: input });
    setInput('');
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Messages List */}
      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          bgcolor: 'background.default',
          p: 2,
        }}
      >
        <List>
          {messages.map((message, index) => (
            <ListItem
              key={message.id}
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: message.role === 'user' ? 'flex-end' : 'flex-start',
                mb: 2,
              }}
            >
              <Paper
                className={`message-bubble ${message.role}-message`}
                elevation={1}
              >
                <ReactMarkdown className="markdown-content">
                  {message.content}
                </ReactMarkdown>
              </Paper>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ mt: 0.5 }}
              >
                {new Date(message.timestamp).toLocaleTimeString()}
              </Typography>
              {index < messages.length - 1 && <Divider sx={{ my: 2 }} />}
            </ListItem>
          ))}
        </List>
        {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
            <CircularProgress size={24} />
          </Box>
        )}
        <div ref={messagesEndRef} />
      </Box>

      {/* Input Area */}
      <Paper
        elevation={3}
        sx={{
          p: 2,
          bgcolor: 'background.paper',
          borderTop: 1,
          borderColor: 'divider',
        }}
      >
        <Box sx={{ display: 'flex', gap: 1 }}>
          <IconButton size="small">
            <AttachFileIcon />
          </IconButton>
          <TextField
            fullWidth
            multiline
            maxRows={4}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type a message..."
            variant="outlined"
            size="small"
          />
          <IconButton
            color="primary"
            onClick={handleSendMessage}
            disabled={!input.trim() || isLoading}
          >
            {isLoading ? <StopIcon /> : <SendIcon />}
          </IconButton>
        </Box>
      </Paper>
    </Box>
  );
};

export default Chat;
