import { createAsyncLocalStorage } from "../../client/components/async-local-storage";
export function getBuiltinRequestContext() {
    const _globalThis = globalThis;
    const ctx = _globalThis[NEXT_REQUEST_CONTEXT_SYMBOL];
    return ctx == null ? void 0 : ctx.get();
}
const NEXT_REQUEST_CONTEXT_SYMBOL = Symbol.for("@next/request-context");
/** "@next/request-context" has a different signature from AsyncLocalStorage,
 * matching [AsyncContext.Variable](https://github.com/tc39/proposal-async-context).
 * We don't need a full AsyncContext adapter here, just having `.get()` is enough
 */ export function createLocalRequestContext() {
    const storage = createAsyncLocalStorage();
    return {
        get: ()=>storage.getStore(),
        run: (value, callback)=>storage.run(value, callback)
    };
}

//# sourceMappingURL=builtin-request-context.js.map