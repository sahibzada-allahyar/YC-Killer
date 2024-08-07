import { google, calendar_v3 } from 'googleapis';
import { BaseSkill } from '../core/base-skill';
import { SkillContext, SkillParameter, SkillResult, AppError } from '../types';

export class CalendarSkill extends BaseSkill {
  public name = 'calendar';
  public description = 'Manage calendar events (create, list, update, delete)';
  public parameters: SkillParameter[] = [
    {
      name: 'action',
      type: 'string',
      description: 'The action to perform (create, list, update, delete)',
      required: true,
    },
    {
      name: 'title',
      type: 'string',
      description: 'Event title',
      required: false,
    },
    {
      name: 'startTime',
      type: 'string',
      description: 'Event start time (ISO string)',
      required: false,
    },
    {
      name: 'endTime',
      type: 'string',
      description: 'Event end time (ISO string)',
      required: false,
    },
    {
      name: 'description',
      type: 'string',
      description: 'Event description',
      required: false,
    },
    {
      name: 'attendees',
      type: 'array',
      description: 'List of attendee email addresses',
      required: false,
    },
    {
      name: 'eventId',
      type: 'string',
      description: 'Event ID (required for update/delete)',
      required: false,
    },
  ];

  private async getCalendarClient(context: SkillContext): Promise<calendar_v3.Calendar> {
    const { user } = context;
    const googleAuth = user.integrations.google;

    if (!googleAuth) {
      throw new AppError(
        'GOOGLE_AUTH_REQUIRED',
        'Google Calendar integration is required for this skill'
      );
    }

    const oauth2Client = new google.auth.OAuth2();
    oauth2Client.setCredentials({
      access_token: googleAuth.accessToken,
      refresh_token: googleAuth.refreshToken,
    });

    return google.calendar({ version: 'v3', auth: oauth2Client });
  }

  public async execute(
    params: Record<string, unknown>,
    context: SkillContext
  ): Promise<SkillResult> {
    try {
      this.validateParameters(params);
      const calendar = await this.getCalendarClient(context);

      switch (params.action as string) {
        case 'create':
          return await this.createEvent(calendar, params);
        case 'list':
          return await this.listEvents(calendar, params);
        case 'update':
          return await this.updateEvent(calendar, params);
        case 'delete':
          return await this.deleteEvent(calendar, params);
        default:
          throw new AppError(
            'INVALID_ACTION',
            `Invalid action: ${params.action}`
          );
      }
    } catch (error) {
      if (error instanceof AppError) {
        return this.createErrorResult(error.message);
      }
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error occurred'
      );
    }
  }

  private async createEvent(
    calendar: calendar_v3.Calendar,
    params: Record<string, unknown>
  ): Promise<SkillResult> {
    if (!params.title || !params.startTime || !params.endTime) {
      throw new AppError(
        'MISSING_PARAMETERS',
        'Title, start time, and end time are required for creating events'
      );
    }

    const event: calendar_v3.Schema$Event = {
      summary: params.title as string,
      description: params.description as string,
      start: {
        dateTime: params.startTime as string,
        timeZone: 'UTC',
      },
      end: {
        dateTime: params.endTime as string,
        timeZone: 'UTC',
      },
    };

    if (params.attendees) {
      event.attendees = (params.attendees as string[]).map(email => ({ email }));
    }

    const response = await calendar.events.insert({
      calendarId: 'primary',
      requestBody: event,
      sendUpdates: 'all',
    });

    return this.createSuccessResult(
      response.data,
      'Event created successfully'
    );
  }

  private async listEvents(
    calendar: calendar_v3.Calendar,
    params: Record<string, unknown>
  ): Promise<SkillResult> {
    const timeMin = params.startTime || new Date().toISOString();
    const timeMax = params.endTime || new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString();

    const response = await calendar.events.list({
      calendarId: 'primary',
      timeMin: timeMin as string,
      timeMax: timeMax as string,
      maxResults: 10,
      singleEvents: true,
      orderBy: 'startTime',
    });

    return this.createSuccessResult(
      response.data.items,
      `Found ${response.data.items?.length || 0} events`
    );
  }

  private async updateEvent(
    calendar: calendar_v3.Calendar,
    params: Record<string, unknown>
  ): Promise<SkillResult> {
    if (!params.eventId) {
      throw new AppError(
        'MISSING_PARAMETERS',
        'Event ID is required for updating events'
      );
    }

    const event: calendar_v3.Schema$Event = {};
    
    if (params.title) event.summary = params.title as string;
    if (params.description) event.description = params.description as string;
    if (params.startTime) {
      event.start = {
        dateTime: params.startTime as string,
        timeZone: 'UTC',
      };
    }
    if (params.endTime) {
      event.end = {
        dateTime: params.endTime as string,
        timeZone: 'UTC',
      };
    }
    if (params.attendees) {
      event.attendees = (params.attendees as string[]).map(email => ({ email }));
    }

    const response = await calendar.events.patch({
      calendarId: 'primary',
      eventId: params.eventId as string,
      requestBody: event,
      sendUpdates: 'all',
    });

    return this.createSuccessResult(
      response.data,
      'Event updated successfully'
    );
  }

  private async deleteEvent(
    calendar: calendar_v3.Calendar,
    params: Record<string, unknown>
  ): Promise<SkillResult> {
    if (!params.eventId) {
      throw new AppError(
        'MISSING_PARAMETERS',
        'Event ID is required for deleting events'
      );
    }

    await calendar.events.delete({
      calendarId: 'primary',
      eventId: params.eventId as string,
      sendUpdates: 'all',
    });

    return this.createSuccessResult(
      null,
      'Event deleted successfully'
    );
  }
}
