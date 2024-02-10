/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
import { CountryListInstance } from "./messaging/country";
export interface MessagingSolution {
}
export interface MessagingListInstance {
    _version: V1;
    _solution: MessagingSolution;
    _uri: string;
    _countries?: CountryListInstance;
    countries: CountryListInstance;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function MessagingListInstance(version: V1): MessagingListInstance;
