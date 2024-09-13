import { PrismaClient } from '@prisma/client';
import { 
  Call, 
  TranscriptSegment, 
  CallType, 
  CallStatus, 
  CallOutcome 
} from '../utils/types';

class DBService {
  private prisma: PrismaClient;

  constructor() {
    this.prisma = new PrismaClient();
  }

  // Call Operations
  async createCall(
    phoneNumber: string,
    type: CallType,
    status: CallStatus = 'ringing'
  ): Promise<string> {
    const call = await this.prisma.call.create({
      data: {
        phoneNumber,
        type,
        status,
        startTime: new Date()
      }
    });
    return call.id;
  }

  async updateCallStatus(
    callId: string,
    status: CallStatus,
    outcome?: CallOutcome
  ): Promise<void> {
    const data: any = { status };
    if (status === 'completed') {
      data.endTime = new Date();
      if (outcome) {
        data.outcome = outcome;
      }
    }

    await this.prisma.call.update({
      where: { id: callId },
      data
    });
  }

  async addTranscriptSegment(
    callId: string,
    speaker: 'user' | 'assistant',
    text: string
  ): Promise<void> {
    await this.prisma.transcriptSegment.create({
      data: {
        callId,
        speaker,
        text,
        timestamp: new Date()
      }
    });
  }

  async updateCallRecording(
    callId: string,
    recordingUrl: string
  ): Promise<void> {
    await this.prisma.call.update({
      where: { id: callId },
      data: { recordingUrl }
    });
  }

  async getCallWithTranscript(callId: string): Promise<Call | null> {
    const call = await this.prisma.call.findUnique({
      where: { id: callId },
      include: {
        transcriptSegments: {
          orderBy: { timestamp: 'asc' }
        }
      }
    });

    if (!call) return null;

    // Convert Prisma model to our type
    return {
      id: call.id,
      type: call.type as CallType,
      status: call.status as CallStatus,
      phoneNumber: call.phoneNumber,
      startTime: call.startTime,
      endTime: call.endTime || undefined,
      outcome: call.outcome as CallOutcome | undefined,
      recordingUrl: call.recordingUrl || undefined,
      transcriptSegments: call.transcriptSegments.map((segment: { 
        id: string;
        callId: string;
        timestamp: Date;
        speaker: string;
        text: string;
      }) => ({
        id: segment.id,
        callId: segment.callId,
        timestamp: segment.timestamp,
        speaker: segment.speaker as 'user' | 'assistant',
        text: segment.text
      }))
    };
  }

  async getRecentCalls(limit: number = 10): Promise<Call[]> {
    const calls = await this.prisma.call.findMany({
      take: limit,
      orderBy: { startTime: 'desc' },
      include: {
        transcriptSegments: true
      }
    });

    return calls.map((call: {
      id: string;
      type: string;
      status: string;
      phoneNumber: string;
      startTime: Date;
      endTime: Date | null;
      outcome: string | null;
      recordingUrl: string | null;
      transcriptSegments: Array<{
        id: string;
        callId: string;
        timestamp: Date;
        speaker: string;
        text: string;
      }>;
    }) => ({
      id: call.id,
      type: call.type as CallType,
      status: call.status as CallStatus,
      phoneNumber: call.phoneNumber,
      startTime: call.startTime,
      endTime: call.endTime || undefined,
      outcome: call.outcome as CallOutcome | undefined,
      recordingUrl: call.recordingUrl || undefined,
      transcriptSegments: call.transcriptSegments.map((segment: {
        id: string;
        callId: string;
        timestamp: Date;
        speaker: string;
        text: string;
      }) => ({
        id: segment.id,
        callId: segment.callId,
        timestamp: segment.timestamp,
        speaker: segment.speaker as 'user' | 'assistant',
        text: segment.text
      }))
    }));
  }

  // Feedback Operations
  async addCallFeedback(
    callId: string,
    reviewerId: string,
    rating: number,
    comments?: string
  ): Promise<void> {
    await this.prisma.callFeedback.create({
      data: {
        callId,
        reviewerId,
        rating,
        comments
      }
    });
  }

  // Training Data Operations
  async saveTrainingExample(
    input: string,
    output: string,
    context: string,
    rating: number,
    feedback?: string,
    callId?: string
  ): Promise<void> {
    await this.prisma.trainingExample.create({
      data: {
        input,
        output,
        context,
        rating,
        feedback,
        callId
      }
    });
  }

  // User Operations
  async findUserByEmail(email: string) {
    return this.prisma.user.findUnique({
      where: { email }
    });
  }

  async createUser(
    email: string,
    name: string,
    role: 'admin' | 'supervisor',
    passwordHash: string
  ) {
    return this.prisma.user.create({
      data: {
        email,
        name,
        role,
        passwordHash
      }
    });
  }

  // Cleanup
  async disconnect() {
    await this.prisma.$disconnect();
  }
}

export const dbService = new DBService();
