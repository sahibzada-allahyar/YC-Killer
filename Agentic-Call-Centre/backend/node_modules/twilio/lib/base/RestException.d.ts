interface RestExceptionError {
    status: number;
    message?: string;
    code?: number;
    moreInfo?: string;
    details?: object;
}
export default class RestException extends Error implements RestExceptionError {
    status: number;
    message: string;
    code?: number;
    moreInfo?: string;
    details?: object;
    constructor(response: any);
}
export {};
