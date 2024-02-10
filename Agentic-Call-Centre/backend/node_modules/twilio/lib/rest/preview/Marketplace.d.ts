import PreviewBase from "../PreviewBase";
import Version from "../../base/Version";
import { AvailableAddOnListInstance } from "./marketplace/availableAddOn";
import { InstalledAddOnListInstance } from "./marketplace/installedAddOn";
export default class Marketplace extends Version {
    /**
     * Initialize the Marketplace version of Preview
     *
     * @param domain - The Twilio (Twilio.Preview) domain
     */
    constructor(domain: PreviewBase);
    /** availableAddOns - { Twilio.Preview.Marketplace.AvailableAddOnListInstance } resource */
    protected _availableAddOns?: AvailableAddOnListInstance;
    /** installedAddOns - { Twilio.Preview.Marketplace.InstalledAddOnListInstance } resource */
    protected _installedAddOns?: InstalledAddOnListInstance;
    /** Getter for availableAddOns resource */
    get availableAddOns(): AvailableAddOnListInstance;
    /** Getter for installedAddOns resource */
    get installedAddOns(): InstalledAddOnListInstance;
}
