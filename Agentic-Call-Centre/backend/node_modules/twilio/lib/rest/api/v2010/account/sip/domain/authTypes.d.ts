/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2010 from "../../../../V2010";
import { AuthTypeCallsListInstance } from "./authTypes/authTypeCalls";
import { AuthTypeRegistrationsListInstance } from "./authTypes/authTypeRegistrations";
export interface AuthTypesSolution {
    accountSid: string;
    domainSid: string;
}
export interface AuthTypesListInstance {
    _version: V2010;
    _solution: AuthTypesSolution;
    _uri: string;
    _calls?: AuthTypeCallsListInstance;
    calls: AuthTypeCallsListInstance;
    _registrations?: AuthTypeRegistrationsListInstance;
    registrations: AuthTypeRegistrationsListInstance;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AuthTypesListInstance(version: V2010, accountSid: string, domainSid: string): AuthTypesListInstance;
