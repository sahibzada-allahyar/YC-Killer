import { Skill, SkillParameter, SkillContext, SkillResult, AppError } from '../types';

export abstract class BaseSkill implements Skill {
  public abstract name: string;
  public abstract description: string;
  public abstract parameters: SkillParameter[];

  protected validateParameters(params: Record<string, unknown>): void {
    for (const param of this.parameters) {
      if (param.required && !(param.name in params)) {
        throw new AppError(
          'INVALID_PARAMETERS',
          `Missing required parameter: ${param.name}`
        );
      }

      if (param.name in params) {
        const value = params[param.name];
        this.validateParameterType(param, value);
      }
    }
  }

  private validateParameterType(param: SkillParameter, value: unknown): void {
    switch (param.type) {
      case 'string':
        if (typeof value !== 'string') {
          throw new AppError(
            'INVALID_PARAMETER_TYPE',
            `Parameter ${param.name} must be a string`
          );
        }
        break;
      case 'number':
        if (typeof value !== 'number') {
          throw new AppError(
            'INVALID_PARAMETER_TYPE',
            `Parameter ${param.name} must be a number`
          );
        }
        break;
      case 'boolean':
        if (typeof value !== 'boolean') {
          throw new AppError(
            'INVALID_PARAMETER_TYPE',
            `Parameter ${param.name} must be a boolean`
          );
        }
        break;
      case 'object':
        if (typeof value !== 'object' || value === null || Array.isArray(value)) {
          throw new AppError(
            'INVALID_PARAMETER_TYPE',
            `Parameter ${param.name} must be an object`
          );
        }
        break;
      case 'array':
        if (!Array.isArray(value)) {
          throw new AppError(
            'INVALID_PARAMETER_TYPE',
            `Parameter ${param.name} must be an array`
          );
        }
        break;
    }
  }

  protected createSuccessResult(data: unknown, message: string): SkillResult {
    return {
      success: true,
      data,
      message,
    };
  }

  protected createErrorResult(error: string): SkillResult {
    return {
      success: false,
      error,
      message: error,
    };
  }

  public abstract execute(
    params: Record<string, unknown>,
    context: SkillContext
  ): Promise<SkillResult>;
}
