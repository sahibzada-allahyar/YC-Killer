'use client';

import { Call } from '@/types';
import {
  PhoneIcon,
  ClockIcon,
  UserIcon,
  ComputerDesktopIcon,
} from '@heroicons/react/24/outline';
import AudioPlayer from '@/components/AudioPlayer';
import { useRef, useEffect } from 'react';

interface CallDetailsProps {
  call: Call;
}

export default function CallDetails({ call }: CallDetailsProps) {
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (call.status === 'in-progress') {
      transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [call.transcriptSegments, call.status]);

  const formatDuration = (startTime: Date, endTime?: Date) => {
    const start = new Date(startTime);
    const end = endTime ? new Date(endTime) : new Date();
    const diff = end.getTime() - start.getTime();
    const minutes = Math.floor(diff / 60000);
    const seconds = Math.floor((diff % 60000) / 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      {/* Call Header */}
      <div className="border-b border-gray-200 pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <PhoneIcon className={`h-6 w-6 ${
              call.type === 'sales' ? 'text-green-500' : 'text-blue-500'
            }`} />
            <h2 className="ml-2 text-lg font-medium text-gray-900">
              {call.phoneNumber}
            </h2>
          </div>
          <span className={`text-sm font-medium px-3 py-1 rounded-full ${
            call.status === 'in-progress'
              ? 'bg-green-100 text-green-800'
              : call.status === 'ringing'
              ? 'bg-yellow-100 text-yellow-800'
              : 'bg-gray-100 text-gray-800'
          }`}>
            {call.status}
          </span>
        </div>
        <div className="mt-2 flex items-center text-sm text-gray-500">
          <ClockIcon className="h-4 w-4 mr-1" />
          <span>Duration: {formatDuration(call.startTime, call.endTime)}</span>
          <span className="mx-2">•</span>
          <span className="capitalize">{call.type}</span>
          {call.outcome && (
            <>
              <span className="mx-2">•</span>
              <span>Outcome: {call.outcome}</span>
            </>
          )}
        </div>
      </div>

      {/* Audio Player */}
      {call.recordingUrl && (
        <div className="mt-4">
          <AudioPlayer
            url={call.recordingUrl}
            onTimeUpdate={(time) => {
              // Find transcript segment closest to current time
              const currentSegment = call.transcriptSegments.find(
                (segment, index) => {
                  const nextSegment = call.transcriptSegments[index + 1];
                  const segmentTime = new Date(segment.timestamp).getTime() / 1000;
                  const nextSegmentTime = nextSegment
                    ? new Date(nextSegment.timestamp).getTime() / 1000
                    : Infinity;
                  return time >= segmentTime && time < nextSegmentTime;
                }
              );

              if (currentSegment) {
                // Scroll to the current segment in transcript
                document.getElementById(currentSegment.id)?.scrollIntoView({
                  behavior: 'smooth',
                  block: 'center'
                });
              }
            }}
          />
        </div>
      )}

      {/* Transcript */}
      <div className="mt-4">
        <div className="bg-gray-50 rounded-lg p-4 h-[500px] overflow-y-auto">
          {call.transcriptSegments.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              Waiting for conversation to begin...
            </div>
          ) : (
            <div className="space-y-4">
              {call.transcriptSegments.map((segment) => (
                <div
                  key={segment.id}
                  id={segment.id}
                  className={`flex items-start space-x-3 mb-4 ${
                    segment.speaker === 'assistant' ? 'flex-row' : 'flex-row-reverse'
                  }`}
                >
                  <div className={`flex-shrink-0 ${
                    segment.speaker === 'assistant' ? 'text-blue-500' : 'text-green-500'
                  }`}>
                    {segment.speaker === 'assistant' ? (
                      <ComputerDesktopIcon className="h-6 w-6" />
                    ) : (
                      <UserIcon className="h-6 w-6" />
                    )}
                  </div>
                  <div className={`flex-1 ${
                    segment.speaker === 'assistant' ? 'mr-12' : 'ml-12'
                  }`}>
                    <div className={`rounded-lg p-3 ${
                      segment.speaker === 'assistant'
                        ? 'bg-blue-50 text-blue-900'
                        : 'bg-green-50 text-green-900'
                    }`}>
                      <p className="text-sm">{segment.text}</p>
                    </div>
                    <span className="text-xs text-gray-500 mt-1 block">
                      {new Date(segment.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              ))}
              <div ref={transcriptEndRef} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
