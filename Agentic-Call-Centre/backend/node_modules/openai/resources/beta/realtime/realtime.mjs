// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.
import { APIResource } from "../../../resource.mjs";
import * as SessionsAPI from "./sessions.mjs";
import { Sessions, } from "./sessions.mjs";
export class Realtime extends APIResource {
    constructor() {
        super(...arguments);
        this.sessions = new SessionsAPI.Sessions(this._client);
    }
}
Realtime.Sessions = Sessions;
//# sourceMappingURL=realtime.mjs.map