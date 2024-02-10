export default class Response<TPayload> {
    statusCode: number;
    body: TPayload;
    headers: any;
    constructor(statusCode: number, body: TPayload, headers: any);
    toString(): string;
}
