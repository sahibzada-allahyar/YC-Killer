'use client';

import { useState, useEffect } from 'react';
import { Call, WebSocketMessage } from '@/types';
import { callsApi } from '@/services/api';
import { useWebSocket } from '@/hooks/useWebSocket';
import CallList from '@/components/calls/CallList';
import CallDetails from '@/components/calls/CallDetails';
import { useCallback } from 'react';

export default function CallsPage() {
  const [calls, setCalls] = useState<Call[]>([]);
  const [selectedCall, setSelectedCall] = useState<Call | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { isConnected, error: wsError } = useWebSocket({
    url: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3000/ws',
    onMessage: (message: WebSocketMessage) => {
      switch (message.type) {
        case 'callStarted':
          setCalls(prev => [message.data as Call, ...prev]);
          break;
        case 'callEnded':
          setCalls(prev =>
            prev.map(call =>
              call.id === (message.data as Call).id
                ? { ...call, status: 'completed', endTime: new Date() }
                : call
            )
          );
          if (selectedCall?.id === (message.data as Call).id) {
            setSelectedCall(prev =>
              prev ? { ...prev, status: 'completed', endTime: new Date() } : null
            );
          }
          break;
        case 'transcriptUpdate':
          const { callId, segment } = message.data as {
            callId: string;
            segment: Call['transcriptSegments'][0];
          };
          setCalls(prev =>
            prev.map(call =>
              call.id === callId
                ? {
                    ...call,
                    transcriptSegments: [...call.transcriptSegments, segment],
                  }
                : call
            )
          );
          if (selectedCall?.id === callId) {
            setSelectedCall(prev =>
              prev
                ? {
                    ...prev,
                    transcriptSegments: [...prev.transcriptSegments, segment],
                  }
                : null
            );
          }
          break;
        case 'statusUpdate':
          const { callId: id, status } = message.data as {
            callId: string;
            status: Call['status'];
          };
          setCalls(prev =>
            prev.map(call =>
              call.id === id ? { ...call, status } : call
            )
          );
          if (selectedCall?.id === id) {
            setSelectedCall(prev =>
              prev ? { ...prev, status } : null
            );
          }
          break;
      }
    },
  });

  useEffect(() => {
    const fetchCalls = async () => {
      try {
        setIsLoading(true);
        const activeCalls = await callsApi.getActiveCalls();
        setCalls(activeCalls);
        setError(null);
      } catch (err) {
        setError('Failed to fetch calls');
        console.error('Error fetching calls:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchCalls();
  }, []);

  const handleCallSelect = async (callId: string) => {
    try {
      const call = await callsApi.getCall(callId);
      setSelectedCall(call);
      setError(null);
    } catch (err) {
      setError('Failed to fetch call details');
      console.error('Error fetching call details:', err);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (error || wsError) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-red-500">{error || wsError}</div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Call Center</h1>
        <div className="mt-2 flex items-center">
          <span
            className={`inline-block w-2 h-2 rounded-full mr-2 ${
              isConnected ? 'bg-green-500' : 'bg-red-500'
            }`}
          />
          <span className="text-sm text-gray-600">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Call List */}
        <div className="lg:col-span-1">
          <CallList
            calls={calls}
            selectedCallId={selectedCall?.id}
            onCallSelect={useCallback((callId: string) => {
              handleCallSelect(callId);
            }, [handleCallSelect])}
          />
        </div>

        {/* Call Details */}
        <div className="lg:col-span-2">
          {selectedCall ? (
            <CallDetails call={selectedCall} />
          ) : (
            <div className="bg-white rounded-lg shadow p-6 text-center">
              <p className="text-gray-500">Select a call to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
