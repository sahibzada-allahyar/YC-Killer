import { Request, Response } from 'express';
import { telephonyService } from '../services/telephonyService';
import { CallType, APIResponse } from '../utils/types';
import { config } from '../config';
import twilio from 'twilio';
import { dbService } from '../services/dbService';

// Twilio webhook authentication middleware
const validateTwilioRequest = twilio.webhook({
  validate: config.nodeEnv === 'production', // Only validate in production
  authToken: config.twilioAuthToken
});

class CallController {
  /**
   * Handle incoming calls
   */
  public handleIncoming = [
    validateTwilioRequest,
    (req: Request, res: Response) => {
      try {
        const callSid = req.body.CallSid;
        // Default to support for incoming calls
        const type: CallType = 'support';
        
        const twiml = telephonyService.handleIncomingCall(callSid, type);
        res.type('text/xml').send(twiml);
      } catch (error) {
        console.error('Error handling incoming call:', error);
        const response = new twilio.twiml.VoiceResponse();
        response.say('We are experiencing technical difficulties. Please try again later.');
        res.type('text/xml').send(response.toString());
      }
    }
  ];

  /**
   * Handle outbound call initiation
   */
  public handleOutbound = [
    validateTwilioRequest,
    (req: Request, res: Response) => {
      try {
        const callSid = req.body.CallSid;
        const type: CallType = req.query.type as CallType || 'sales';
        
        const twiml = telephonyService.handleIncomingCall(callSid, type);
        res.type('text/xml').send(twiml);
      } catch (error) {
        console.error('Error handling outbound call:', error);
        const response = new twilio.twiml.VoiceResponse();
        response.say('We are experiencing technical difficulties. Please try again later.');
        res.type('text/xml').send(response.toString());
      }
    }
  ];

  /**
   * Handle speech recognition results and generate AI response
   */
  public handleResponse = [
    validateTwilioRequest,
    async (req: Request, res: Response) => {
      try {
        const callSid = req.body.CallSid;
        const userMessage = req.body.SpeechResult;

        if (!userMessage) {
          // If no speech was detected, reprompt
          const response = new twilio.twiml.VoiceResponse();
          response.gather({
            input: ['speech'],
            action: '/api/call/response',
            timeout: 3,
            speechTimeout: 'auto',
            language: 'en-US',
            enhanced: true
          }).say("I didn't catch that. Could you please repeat what you said?");
          return res.type('text/xml').send(response.toString());
        }

        const twiml = await telephonyService.generateResponseTwiML(callSid, userMessage);
        res.type('text/xml').send(twiml);
      } catch (error) {
        console.error('Error handling response:', error);
        const response = new twilio.twiml.VoiceResponse();
        response.say('I apologize, but I encountered an error. Please try again.');
        response.redirect({ method: 'POST' }, '/api/call/response');
        res.type('text/xml').send(response.toString());
      }
    }
  ];

  /**
   * Handle call completion
   */
  public handleComplete = [
    validateTwilioRequest,
    async (req: Request, res: Response) => {
      try {
        const callSid = req.body.CallSid;
        await telephonyService.handleCallComplete(callSid);
        res.sendStatus(200);
      } catch (error) {
        console.error('Error handling call completion:', error);
        res.sendStatus(500);
      }
    }
  ];

  /**
   * Handle recording status updates
   */
  public handleRecordingStatus = [
    validateTwilioRequest,
    async (req: Request, res: Response) => {
      try {
        const callSid = req.body.CallSid;
        const recordingUrl = req.body.RecordingUrl;
        const status = req.body.RecordingStatus;

        if (status === 'completed' && recordingUrl) {
          await dbService.updateCallRecording(callSid, recordingUrl);
        }

        res.sendStatus(200);
      } catch (error) {
        console.error('Error handling recording status:', error);
        res.sendStatus(500);
      }
    }
  ];

  /**
   * Initiate an outbound call (REST API endpoint)
   */
  public initiateOutboundCall = async (req: Request, res: Response) => {
    try {
      const { phoneNumber, type = 'sales' } = req.body;

      if (!phoneNumber) {
        const response: APIResponse<null> = {
          success: false,
          error: 'Phone number is required'
        };
        return res.status(400).json(response);
      }

      const callSid = await telephonyService.makeOutboundCall(phoneNumber, type as CallType);
      
      const response: APIResponse<{ callSid: string }> = {
        success: true,
        data: { callSid }
      };
      res.json(response);
    } catch (error) {
      console.error('Error initiating outbound call:', error);
      const response: APIResponse<null> = {
        success: false,
        error: 'Failed to initiate call'
      };
      res.status(500).json(response);
    }
  };

  /**
   * Get call recording (REST API endpoint)
   */
  public getRecording = async (req: Request, res: Response) => {
    try {
      const callSid = req.params.callSid;
      const recordingUrl = await telephonyService.getCallRecording(callSid);

      if (!recordingUrl) {
        const response: APIResponse<null> = {
          success: false,
          error: 'Recording not found'
        };
        return res.status(404).json(response);
      }

      const response: APIResponse<{ url: string }> = {
        success: true,
        data: { url: recordingUrl }
      };
      res.json(response);
    } catch (error) {
      console.error('Error fetching recording:', error);
      const response: APIResponse<null> = {
        success: false,
        error: 'Failed to fetch recording'
      };
      res.status(500).json(response);
    }
  };
}

export const callController = new CallController();
