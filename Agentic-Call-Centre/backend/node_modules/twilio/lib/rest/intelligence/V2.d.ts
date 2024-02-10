import IntelligenceBase from "../IntelligenceBase";
import Version from "../../base/Version";
import { ServiceListInstance } from "./v2/service";
import { TranscriptListInstance } from "./v2/transcript";
export default class V2 extends Version {
    /**
     * Initialize the V2 version of Intelligence
     *
     * @param domain - The Twilio (Twilio.Intelligence) domain
     */
    constructor(domain: IntelligenceBase);
    /** services - { Twilio.Intelligence.V2.ServiceListInstance } resource */
    protected _services?: ServiceListInstance;
    /** transcripts - { Twilio.Intelligence.V2.TranscriptListInstance } resource */
    protected _transcripts?: TranscriptListInstance;
    /** Getter for services resource */
    get services(): ServiceListInstance;
    /** Getter for transcripts resource */
    get transcripts(): TranscriptListInstance;
}
