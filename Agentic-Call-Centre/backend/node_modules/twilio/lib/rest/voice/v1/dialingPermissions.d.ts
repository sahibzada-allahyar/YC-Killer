/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
import { BulkCountryUpdateListInstance } from "./dialingPermissions/bulkCountryUpdate";
import { CountryListInstance } from "./dialingPermissions/country";
import { SettingsListInstance } from "./dialingPermissions/settings";
export interface DialingPermissionsSolution {
}
export interface DialingPermissionsListInstance {
    _version: V1;
    _solution: DialingPermissionsSolution;
    _uri: string;
    _bulkCountryUpdates?: BulkCountryUpdateListInstance;
    bulkCountryUpdates: BulkCountryUpdateListInstance;
    _countries?: CountryListInstance;
    countries: CountryListInstance;
    _settings?: SettingsListInstance;
    settings: SettingsListInstance;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function DialingPermissionsListInstance(version: V1): DialingPermissionsListInstance;
