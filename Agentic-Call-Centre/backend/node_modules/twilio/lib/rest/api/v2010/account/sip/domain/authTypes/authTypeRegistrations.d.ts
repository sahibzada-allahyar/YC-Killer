/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2010 from "../../../../../V2010";
import { AuthRegistrationsCredentialListMappingListInstance } from "./authTypeRegistrations/authRegistrationsCredentialListMapping";
export interface AuthTypeRegistrationsSolution {
    accountSid: string;
    domainSid: string;
}
export interface AuthTypeRegistrationsListInstance {
    _version: V2010;
    _solution: AuthTypeRegistrationsSolution;
    _uri: string;
    _credentialListMappings?: AuthRegistrationsCredentialListMappingListInstance;
    credentialListMappings: AuthRegistrationsCredentialListMappingListInstance;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AuthTypeRegistrationsListInstance(version: V2010, accountSid: string, domainSid: string): AuthTypeRegistrationsListInstance;
