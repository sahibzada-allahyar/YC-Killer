import { Call } from '@/types';
import {
  PhoneIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';

interface CallListProps {
  calls: Call[];
  selectedCallId?: string;
  onCallSelect: (callId: string) => void;
}

export default function CallList({ calls, selectedCallId, onCallSelect }: CallListProps) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h2 className="text-lg font-medium text-gray-900 mb-4">Active Calls</h2>
      <div className="space-y-4">
        {calls.length === 0 ? (
          <p className="text-gray-500 text-center py-4">
            No active calls
          </p>
        ) : (
          calls.map((call) => (
            <button
              key={call.id}
              onClick={() => onCallSelect(call.id)}
              className={`w-full text-left p-4 rounded-lg border transition-colors ${
                selectedCallId === call.id
                  ? 'border-primary-500 bg-primary-50'
                  : 'border-gray-200 hover:bg-gray-50'
              }`}
            >
              <div className="flex justify-between items-start">
                <div>
                  <p className="font-medium text-gray-900">
                    {call.phoneNumber}
                  </p>
                  <p className="text-sm text-gray-500">
                    {new Date(call.startTime).toLocaleString()}
                  </p>
                </div>
                <span
                  className={`text-sm font-medium px-2 py-1 rounded-full ${
                    call.type === 'sales'
                      ? 'bg-green-100 text-green-800'
                      : 'bg-blue-100 text-blue-800'
                  }`}
                >
                  {call.type}
                </span>
              </div>
              <div className="mt-2 flex items-center text-sm text-gray-500">
                <PhoneIcon className="h-4 w-4 mr-1" />
                <span className="capitalize">{call.status}</span>
                {call.endTime && (
                  <>
                    <span className="mx-2">•</span>
                    <ClockIcon className="h-4 w-4 mr-1" />
                    <span>
                      {Math.round(
                        (new Date(call.endTime).getTime() -
                          new Date(call.startTime).getTime()) /
                          60000
                      )}m
                    </span>
                  </>
                )}
              </div>
              {call.outcome && (
                <p className="mt-2 text-sm text-gray-600">
                  Outcome: {call.outcome}
                </p>
              )}
            </button>
          ))
        )}
      </div>
    </div>
  );
}
