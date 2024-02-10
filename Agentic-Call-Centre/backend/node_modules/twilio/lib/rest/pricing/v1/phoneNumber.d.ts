/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
import { CountryListInstance } from "./phoneNumber/country";
export interface PhoneNumberSolution {
}
export interface PhoneNumberListInstance {
    _version: V1;
    _solution: PhoneNumberSolution;
    _uri: string;
    _countries?: CountryListInstance;
    countries: CountryListInstance;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function PhoneNumberListInstance(version: V1): PhoneNumberListInstance;
