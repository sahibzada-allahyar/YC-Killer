import { APIResource } from "../../../resource.js";
import * as Core from "../../../core.js";
import * as CompletionsAPI from "./completions.js";
import { ChatCompletionStoreMessagesPage } from "./completions.js";
import { type CursorPageParams } from "../../../pagination.js";
export declare class Messages extends APIResource {
    /**
     * Get the messages in a stored chat completion. Only chat completions that have
     * been created with the `store` parameter set to `true` will be returned.
     */
    list(completionId: string, query?: MessageListParams, options?: Core.RequestOptions): Core.PagePromise<ChatCompletionStoreMessagesPage, CompletionsAPI.ChatCompletionStoreMessage>;
    list(completionId: string, options?: Core.RequestOptions): Core.PagePromise<ChatCompletionStoreMessagesPage, CompletionsAPI.ChatCompletionStoreMessage>;
}
export interface MessageListParams extends CursorPageParams {
    /**
     * Sort order for messages by timestamp. Use `asc` for ascending order or `desc`
     * for descending order. Defaults to `asc`.
     */
    order?: 'asc' | 'desc';
}
export declare namespace Messages {
    export { type MessageListParams as MessageListParams };
}
export { ChatCompletionStoreMessagesPage };
//# sourceMappingURL=messages.d.ts.map