import ContentBase from "../ContentBase";
import Version from "../../base/Version";
import { ContentListInstance } from "./v1/content";
import { ContentAndApprovalsListInstance } from "./v1/contentAndApprovals";
import { LegacyContentListInstance } from "./v1/legacyContent";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Content
     *
     * @param domain - The Twilio (Twilio.Content) domain
     */
    constructor(domain: ContentBase);
    /** contents - { Twilio.Content.V1.ContentListInstance } resource */
    protected _contents?: ContentListInstance;
    /** contentAndApprovals - { Twilio.Content.V1.ContentAndApprovalsListInstance } resource */
    protected _contentAndApprovals?: ContentAndApprovalsListInstance;
    /** legacyContents - { Twilio.Content.V1.LegacyContentListInstance } resource */
    protected _legacyContents?: LegacyContentListInstance;
    /** Getter for contents resource */
    get contents(): ContentListInstance;
    /** Getter for contentAndApprovals resource */
    get contentAndApprovals(): ContentAndApprovalsListInstance;
    /** Getter for legacyContents resource */
    get legacyContents(): LegacyContentListInstance;
}
