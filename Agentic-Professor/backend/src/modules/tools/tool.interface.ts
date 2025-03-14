export interface ITool {
  name: string;
  description: string;
  parameters: {
    properties: Record<string, unknown>;
    required?: string[];
  };
  execute(input: string): Promise<unknown>;
}
