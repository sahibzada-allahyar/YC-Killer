'use client';

import { useState, useEffect } from 'react';
import { Call, CallFeedback } from '@/types';
import { callsApi } from '@/services/api';
import AudioPlayer from '@/components/AudioPlayer';
import {
  StarIcon,
  CheckCircleIcon,
  XCircleIcon,
} from '@heroicons/react/24/solid';
import { StarIcon as StarOutlineIcon } from '@heroicons/react/24/outline';

export default function TrainingPage() {
  const [calls, setCalls] = useState<Call[]>([]);
  const [selectedCall, setSelectedCall] = useState<Call | null>(null);
  const [feedback, setFeedback] = useState<Partial<CallFeedback>>({
    rating: 0,
    comments: '',
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  useEffect(() => {
    // Fetch completed calls that need review
    const fetchCalls = async () => {
      try {
        const response = await callsApi.getRecentCalls();
        setCalls(response.filter(call => call.status === 'completed'));
      } catch (err) {
        setError('Failed to fetch calls');
        console.error('Error fetching calls:', err);
      }
    };

    fetchCalls();
  }, []);

  const handleSubmitFeedback = async () => {
    if (!selectedCall || !feedback.rating) {
      setError('Please provide a rating');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      await callsApi.submitFeedback(selectedCall.id, {
        reviewerId: 'current-user-id', // This should come from auth context
        rating: feedback.rating,
        comments: feedback.comments,
      });

      setSuccessMessage('Feedback submitted successfully');
      
      // Remove the reviewed call from the list
      setCalls(calls.filter(call => call.id !== selectedCall.id));
      setSelectedCall(null);
      setFeedback({ rating: 0, comments: '' });

      // Clear success message after 3 seconds
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError('Failed to submit feedback');
      console.error('Error submitting feedback:', err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Training Dashboard</h1>
        <p className="mt-2 text-gray-600">
          Review calls and provide feedback to improve the AI agent
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Call List */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="text-lg font-medium text-gray-900 mb-4">
              Calls to Review
            </h2>
            <div className="space-y-4">
              {calls.length === 0 ? (
                <p className="text-gray-500 text-center py-4">
                  No calls available for review
                </p>
              ) : (
                calls.map((call) => (
                  <button
                    key={call.id}
                    onClick={() => {
                      setSelectedCall(call);
                      setFeedback({ rating: 0, comments: '' });
                      setError(null);
                    }}
                    className={`w-full text-left p-4 rounded-lg border transition-colors ${
                      selectedCall?.id === call.id
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
        </div>

        {/* Call Review */}
        <div className="lg:col-span-2">
          {selectedCall ? (
            <div className="bg-white rounded-lg shadow p-6">
              <div className="mb-6">
                <h3 className="text-lg font-medium text-gray-900">
                  Review Call
                </h3>
                <p className="text-sm text-gray-500 mt-1">
                  Listen to the call and provide feedback
                </p>
              </div>

              {/* Audio Player */}
              {selectedCall.recordingUrl && (
                <div className="mb-6">
                  <AudioPlayer url={selectedCall.recordingUrl} />
                </div>
              )}

              {/* Transcript */}
              <div className="mb-6">
                <h4 className="text-md font-medium text-gray-900 mb-2">
                  Transcript
                </h4>
                <div className="bg-gray-50 rounded-lg p-4 h-[200px] overflow-y-auto">
                  {selectedCall.transcriptSegments.map((segment) => (
                    <div key={segment.id} className="mb-3">
                      <p className="text-sm font-medium text-gray-700">
                        {segment.speaker === 'assistant' ? 'AI Agent' : 'Customer'}:
                      </p>
                      <p className="text-sm text-gray-600 ml-4">
                        {segment.text}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Feedback Form */}
              <div>
                <h4 className="text-md font-medium text-gray-900 mb-4">
                  Your Feedback
                </h4>

                {/* Rating */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Rate the AI's Performance
                  </label>
                  <div className="flex space-x-2">
                    {[1, 2, 3, 4, 5].map((rating) => (
                      <button
                        key={rating}
                        onClick={() => setFeedback({ ...feedback, rating })}
                        className="focus:outline-none"
                      >
                        {rating <= (feedback.rating || 0) ? (
                          <StarIcon className="h-8 w-8 text-yellow-400" />
                        ) : (
                          <StarOutlineIcon className="h-8 w-8 text-gray-300" />
                        )}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Comments */}
                <div className="mb-6">
                  <label
                    htmlFor="comments"
                    className="block text-sm font-medium text-gray-700 mb-2"
                  >
                    Additional Comments
                  </label>
                  <textarea
                    id="comments"
                    rows={4}
                    value={feedback.comments}
                    onChange={(e) =>
                      setFeedback({ ...feedback, comments: e.target.value })
                    }
                    className="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
                    placeholder="What did the AI do well? What could be improved?"
                  />
                </div>

                {/* Error/Success Messages */}
                {error && (
                  <div className="mb-4 flex items-center text-red-600">
                    <XCircleIcon className="h-5 w-5 mr-2" />
                    {error}
                  </div>
                )}
                {successMessage && (
                  <div className="mb-4 flex items-center text-green-600">
                    <CheckCircleIcon className="h-5 w-5 mr-2" />
                    {successMessage}
                  </div>
                )}

                {/* Submit Button */}
                <button
                  onClick={handleSubmitFeedback}
                  disabled={isSubmitting}
                  className={`w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 ${
                    isSubmitting ? 'opacity-75 cursor-not-allowed' : ''
                  }`}
                >
                  {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
                </button>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow p-6 text-center">
              <p className="text-gray-500">
                Select a call from the list to review
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
