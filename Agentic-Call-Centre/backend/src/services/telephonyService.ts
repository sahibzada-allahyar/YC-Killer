import twilio, { Twilio, twiml } from 'twilio';
import { config } from '../config';
import { CallType, CallStatus, Call, TranscriptSegment } from '../utils/types';
import { llmService } from './llmService';
import { dbService } from './dbService';

class TelephonyService {
  private client: Twilio;
  private activeCallStates: Map<string, {
    type: CallType;
    transcript: TranscriptSegment[];
  }>;

  constructor() {
    this.client = new Twilio(config.twilioAccountSid, config.twilioAuthToken);
    this.activeCallStates = new Map();
  }

  /**
   * Initialize a new call state
   */
  private initializeCallState(callSid: string, type: CallType) {
    this.activeCallStates.set(callSid, {
      type,
      transcript: []
    });
  }

  /**
   * Get TwiML for greeting the caller
   */
  public generateWelcomeTwiML(callType: CallType): string {
    const response = new twiml.VoiceResponse();
    
    // Start recording the call
    response.record({
      action: '/api/call/recording-complete',
      recordingStatusCallback: '/api/call/recording-status',
      recordingStatusCallbackEvent: ['completed'],
    });

    // Initial greeting
    const greeting = callType === 'sales' 
      ? "Hello! Thank you for taking my call. I'm an AI sales representative. How may I assist you today?"
      : "Hello! Thank you for calling our support line. I'm an AI assistant. How can I help you today?";

    // Gather speech input
    response.gather({
      input: ['speech'],
      action: '/api/call/response',
      speechTimeout: 'auto',
      speechModel: 'experimental_conversations'
    }).say(greeting);

    return response.toString();
  }

  /**
   * Generate TwiML for continuing the conversation
   */
  public async generateResponseTwiML(
    callSid: string,
    userMessage: string
  ): Promise<string> {
    const callState = this.activeCallStates.get(callSid);
    if (!callState) {
      throw new Error(`No active call state found for SID: ${callSid}`);
    }

    try {
      // Add user message to transcript
      const userSegment: TranscriptSegment = {
        id: `${callSid}-${callState.transcript.length}`,
        callId: callSid,
        timestamp: new Date(),
        speaker: 'user',
        text: userMessage
      };
      callState.transcript.push(userSegment);

      // Get AI response
      const llmResponse = await llmService.generateResponse(
        userMessage,
        callState.type,
        callState.transcript
      );

      // Add AI response to transcript
      const aiSegment: TranscriptSegment = {
        id: `${callSid}-${callState.transcript.length}`,
        callId: callSid,
        timestamp: new Date(),
        speaker: 'assistant',
        text: llmResponse.text
      };
      callState.transcript.push(aiSegment);

      // Generate TwiML response
      const response = new twiml.VoiceResponse();
      response.gather({
      input: ['speech'],
      action: '/api/call/response',
      timeout: 3,
      speechTimeout: 'auto',
      language: 'en-US',
      enhanced: true
      }).say(llmResponse.text);

      return response.toString();
    } catch (error) {
      console.error('Error generating response:', error);
      
      // Fallback response in case of error
      const response = new twiml.VoiceResponse();
      response.gather({
      input: ['speech'],
      action: '/api/call/response',
      timeout: 3,
      speechTimeout: 'auto',
      language: 'en-US',
      enhanced: true
      }).say("I apologize, but I'm having trouble processing that. Could you please repeat what you said?");
      
      return response.toString();
    }
  }

  /**
   * Initiate an outbound call
   */
  public async makeOutboundCall(
    to: string,
    type: CallType = 'sales'
  ): Promise<string> {
    try {
      const call = await this.client.calls.create({
        to,
        from: config.twilioPhoneNumber,
        url: `${config.corsOrigin}/api/call/outbound?type=${type}`,
        record: true,
        recordingStatusCallback: '/api/call/recording-status',
        recordingStatusCallbackEvent: ['completed']
      });

      this.initializeCallState(call.sid, type);
      return call.sid;
    } catch (error) {
      console.error('Error making outbound call:', error);
      throw new Error('Failed to initiate outbound call');
    }
  }

  /**
   * Handle an incoming call
   */
  public handleIncomingCall(callSid: string, type: CallType = 'support'): string {
    this.initializeCallState(callSid, type);
    return this.generateWelcomeTwiML(type);
  }

  /**
   * Get the transcript for a call
   */
  public getCallTranscript(callSid: string): TranscriptSegment[] {
    const callState = this.activeCallStates.get(callSid);
    if (!callState) {
      throw new Error(`No transcript found for call: ${callSid}`);
    }
    return callState.transcript;
  }

  /**
   * Handle call completion
   */
  public async handleCallComplete(callSid: string): Promise<void> {
    const callState = this.activeCallStates.get(callSid);
    if (!callState) {
      console.warn(`Call ${callSid} completed but no state found`);
      return;
    }

    try {
      // Classify the call outcome
      const outcome = await llmService.classifyCallOutcome(
        callState.transcript,
        callState.type
      );

      // Generate call summary
      const summary = await llmService.generateSummary(callState.transcript);

      // Save call data to database
      await dbService.updateCallStatus(callSid, 'completed', outcome);

      // Clean up call state
      this.activeCallStates.delete(callSid);
    } catch (error) {
      console.error(`Error handling call completion for ${callSid}:`, error);
      throw error;
    }
  }

  /**
   * Get call recording URL
   */
  public async getCallRecording(callSid: string): Promise<string | null> {
    try {
      const recordings = await this.client.recordings.list({ callSid });
      return recordings[0]?.mediaUrl || null;
    } catch (error) {
      console.error('Error fetching recording:', error);
      return null;
    }
  }

  /**
   * Get list of currently active calls
   */
  public getActiveCalls(): { callId: string; type: CallType }[] {
    return Array.from(this.activeCallStates.entries()).map(([callId, state]) => ({
      callId,
      type: state.type
    }));
  }
}

export const telephonyService = new TelephonyService();
