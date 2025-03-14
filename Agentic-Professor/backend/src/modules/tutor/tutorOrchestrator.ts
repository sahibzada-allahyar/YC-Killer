import {
  Config,
  TutorState,
  LLMResponse,
  TutorResponse,
  HomeworkSubmission,
  ProcessedHomework,
  ToolError,
  ToolErrorCode,
  ToolDefinition,
  mergeConfig,
  DEFAULT_CONFIG,
  ToolCall
} from '@ai-tutor/shared';
import { LLMAgent } from '../llm/llmAgent';
import { VoiceAgent } from '../voice/voiceAgent';
import { WikipediaTool } from '../tools/wikipedia/wikipediaTool';
import { ImageProcessingTool } from '../tools/imageProcessing/imageProcessingTool';

export class TutorOrchestrator {
  private llmAgent: LLMAgent;
  private voiceAgent: VoiceAgent;
  private config: Config;
  private state: TutorState;
  private apiKey: string;

  constructor(apiKey: string, userConfig: Partial<Config>) {
    this.apiKey = apiKey;
    
    // Initialize config with default values and merge with provided config
    const defaultConfigCopy: Config = {
      ...DEFAULT_CONFIG,
      tools: [...DEFAULT_CONFIG.tools]
    };
    this.config = mergeConfig(defaultConfigCopy, userConfig);

    // Initialize agents with their specific configurations
    this.llmAgent = new LLMAgent(apiKey, this.config.llm);
    this.voiceAgent = new VoiceAgent(apiKey, this.config);
    
    this.state = {
      activeTools: [],
      processingState: 'idle'
    };

    this.initializeTools();
  }

  private async initializeTools(): Promise<void> {
    // Initialize and attach tools to LLM agent
    if (this.config.enabledFeatures.wikipedia) {
      const wikiTool = new WikipediaTool();
      this.llmAgent.attachTool(wikiTool);
      this.state.activeTools.push('wikipedia');

      const toolDef: ToolDefinition = {
        name: 'wikipedia',
        description: 'Search Wikipedia for articles and information',
        parameters: [{
          type: 'object',
          description: 'Wikipedia search parameters',
          required: true,
          schema: wikiTool.parameters.properties
        }]
      };
      this.config.tools.push(toolDef);
    }

    if (this.config.enabledFeatures.imageProcessing) {
      const imageTool = new ImageProcessingTool({
        defaultLanguage: 'eng'
      });
      this.llmAgent.attachTool(imageTool);
      this.state.activeTools.push('imageProcessing');

      const toolDef: ToolDefinition = {
        name: 'imageProcessing',
        description: 'Process and extract text from images using OCR',
        parameters: [{
          type: 'object',
          description: 'Image processing parameters',
          required: true,
          schema: imageTool.parameters.properties
        }]
      };
      this.config.tools.push(toolDef);
    }
  }

  public async processHomework(submission: HomeworkSubmission): Promise<TutorResponse> {
    try {
      this.state.processingState = 'processing';
      this.state.currentSession = submission.studentId;

      // Process image if provided
      let processedHomework: ProcessedHomework | undefined;
      if (submission.imageData) {
        const imageTool = this.llmAgent.getAttachedTools().find(tool => tool === 'imageProcessing');
        if (!imageTool) {
          throw new ToolError(
            'Image processing is not enabled',
            'imageProcessing',
            ToolErrorCode.INITIALIZATION_FAILED
          );
        }

        // Process the image using the image processing tool
        const result = await this.llmAgent.processQuery(JSON.stringify({
          toolName: 'imageProcessing',
          input: {
            imageData: submission.imageData,
            language: 'eng',
            enhanceImage: true
          }
        }));

        if (result.toolCalls?.[0]?.error) {
          throw new Error(result.toolCalls[0].error);
        }

        processedHomework = {
          extractedText: result.toolCalls?.[0]?.result || '',
          confidence: 0.0 // Extract from tool result
        };
      }

      // Generate tutor response
      const prompt = this.buildPrompt(submission, processedHomework);
      const llmResponse = await this.llmAgent.processQuery(prompt);
      this.state.lastResponse = llmResponse;

      // Generate voice response if enabled
      let audioResponse;
      if (this.config.enabledFeatures.voice) {
        this.state.processingState = 'speaking';
        audioResponse = await this.voiceAgent.speak(llmResponse.content);
      }

      this.state.processingState = 'idle';

      // Construct final response
      const response: TutorResponse = {
        htmlContent: this.formatToHTML(llmResponse),
        plainText: llmResponse.content
      };

      if (audioResponse?.audioUrl) {
        response.audioUrl = audioResponse.audioUrl;
      }

      if (llmResponse.toolCalls?.length) {
        response.references = llmResponse.toolCalls
          .filter((call: ToolCall) => call.toolName === 'wikipedia' && !call.error)
          .map((call: ToolCall) => {
            const result = JSON.parse(call.result || '{}');
            return result.url || '';
          })
          .filter(Boolean);
      }

      return response;
    } catch (error) {
      this.state.processingState = 'error';
      this.state.error = error instanceof Error ? error.message : 'Unknown error occurred';
      throw error;
    }
  }

  private buildPrompt(submission: HomeworkSubmission, processed?: ProcessedHomework): string {
    let prompt = 'You are a helpful tutor. ';

    if (processed?.extractedText) {
      prompt += `The student has submitted homework with the following content:\n\n${processed.extractedText}\n\n`;
    }

    if (submission.subject) {
      prompt += `The subject is ${submission.subject}. `;
    }

    if (submission.additionalNotes) {
      prompt += `Additional context: ${submission.additionalNotes}\n\n`;
    }

    prompt += `
Please provide a detailed explanation that:
1. Identifies the key concepts involved
2. Explains the solution step by step
3. Uses appropriate mathematical notation when needed
4. Includes relevant examples
5. References external sources when helpful
6. Uses diagrams or illustrations where appropriate

Format your response using HTML for rich text formatting, and LaTeX for mathematical equations.`;

    return prompt;
  }

  private formatToHTML(response: LLMResponse): string {
    // Convert the response to HTML with LaTeX equations
    let html = response.content;

    // Convert markdown-style formatting to HTML
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    html = html.replace(/\n\n/g, '</p><p>');

    // Wrap LaTeX equations
    html = html.replace(/\$(.*?)\$/g, '<span class="math">$1</span>');
    html = html.replace(/\$\$(.*?)\$\$/g, '<div class="math">$1</div>');

    // Add code blocks
    html = html.replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>');

    // Wrap in container
    html = `<div class="tutor-response"><p>${html}</p></div>`;

    return html;
  }

  public getState(): TutorState {
    return { ...this.state };
  }

  public async cleanup(): Promise<void> {
    await this.voiceAgent.cleanup();
    // Add any other cleanup needed
  }
}
