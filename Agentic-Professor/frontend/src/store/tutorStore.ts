import { create } from 'zustand';
import { ApiResponse, HomeworkSubmission, TutorResponse } from '@ai-tutor/shared';

interface TutorState {
  isLoading: boolean;
  error: string | null;
  currentResponse: TutorResponse | null;
  submitHomework: (submission: HomeworkSubmission) => Promise<void>;
}

export const useStore = create<TutorState>((set) => ({
  isLoading: false,
  error: null,
  currentResponse: null,

  submitHomework: async (submission: HomeworkSubmission) => {
    try {
      set({ isLoading: true, error: null });

      const response = await fetch('http://localhost:3000/api/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(submission),
      });

      const data = (await response.json()) as ApiResponse<TutorResponse>;

      if (!response.ok || !data.success) {
        throw new Error(data.error?.message || 'Failed to process homework');
      }

      set({
        currentResponse: data.data || null,
        error: null,
      });
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'An unexpected error occurred',
        currentResponse: null,
      });
      throw err;
    } finally {
      set({ isLoading: false });
    }
  },
}));
