import { HttpMethod } from "../interfaces";
export interface RequestOptions<TData> {
    method?: HttpMethod | "*";
    url?: string;
    auth?: string;
    params?: object | "*";
    data?: TData | "*";
    headers?: Headers;
}
export interface Headers {
    [header: string]: string;
}
export default class Request<TData> {
    method: HttpMethod | "*";
    url: string;
    auth: string;
    params: object | "*";
    data: TData | "*";
    headers: Headers | "*";
    constructor(opts?: RequestOptions<TData>);
    get ANY(): "*";
    attributeEqual(lhs: any, rhs: any): boolean;
    isEqual(other: Request<any>): boolean;
    toString(): string;
}
