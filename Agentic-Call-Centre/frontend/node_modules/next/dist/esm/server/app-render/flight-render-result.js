import { RSC_CONTENT_TYPE_HEADER } from "../../client/components/app-router-headers";
import RenderResult from "../render-result";
/**
 * Flight Response is always set to RSC_CONTENT_TYPE_HEADER to ensure it does not get interpreted as HTML.
 */ export class FlightRenderResult extends RenderResult {
    constructor(response, options){
        super(response, {
            contentType: RSC_CONTENT_TYPE_HEADER,
            waitUntil: options == null ? void 0 : options.waitUntil,
            metadata: (options == null ? void 0 : options.metadata) ?? {}
        });
    }
}

//# sourceMappingURL=flight-render-result.js.map