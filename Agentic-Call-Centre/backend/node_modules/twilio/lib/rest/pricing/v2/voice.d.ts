/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../V2";
import { CountryListInstance } from "./voice/country";
import { NumberListInstance } from "./voice/number";
export interface VoiceSolution {
}
export interface VoiceListInstance {
    _version: V2;
    _solution: VoiceSolution;
    _uri: string;
    _countries?: CountryListInstance;
    countries: CountryListInstance;
    _numbers?: NumberListInstance;
    numbers: NumberListInstance;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function VoiceListInstance(version: V2): VoiceListInstance;
