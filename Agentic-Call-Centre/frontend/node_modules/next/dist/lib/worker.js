"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
Object.defineProperty(exports, "Worker", {
    enumerable: true,
    get: function() {
        return Worker;
    }
});
const _jestworker = require("next/dist/compiled/jest-worker");
const RESTARTED = Symbol("restarted");
const cleanupWorkers = (worker)=>{
    var _worker__workerPool;
    for (const curWorker of ((_worker__workerPool = worker._workerPool) == null ? void 0 : _worker__workerPool._workers) || []){
        var _curWorker__child;
        (_curWorker__child = curWorker._child) == null ? void 0 : _curWorker__child.kill("SIGINT");
    }
};
class Worker {
    constructor(workerPath, options){
        this._restarting = false;
        let { timeout, onRestart, logger = console, ...farmOptions } = options;
        let restartPromise;
        let resolveRestartPromise;
        let activeTasks = 0;
        this._worker = undefined;
        // ensure we end workers if they weren't before exit
        process.on("exit", ()=>{
            this.close();
        });
        const createWorker = ()=>{
            var _farmOptions_forkOptions;
            this._worker = new _jestworker.Worker(workerPath, {
                ...farmOptions,
                forkOptions: {
                    ...farmOptions.forkOptions,
                    env: {
                        ...((_farmOptions_forkOptions = farmOptions.forkOptions) == null ? void 0 : _farmOptions_forkOptions.env) || {},
                        ...process.env,
                        IS_NEXT_WORKER: "true"
                    }
                }
            });
            this._restarting = false;
            restartPromise = new Promise((resolve)=>resolveRestartPromise = resolve);
            /**
       * Jest Worker has two worker types, ChildProcessWorker (uses child_process) and NodeThreadWorker (uses worker_threads)
       * Next.js uses ChildProcessWorker by default, but it can be switched to NodeThreadWorker with an experimental flag
       *
       * We only want to handle ChildProcessWorker's orphan process issue, so we access the private property "_child":
       * https://github.com/facebook/jest/blob/b38d7d345a81d97d1dc3b68b8458b1837fbf19be/packages/jest-worker/src/workers/ChildProcessWorker.ts
       *
       * But this property is not available in NodeThreadWorker, so we need to check if we are using ChildProcessWorker
       */ if (!farmOptions.enableWorkerThreads) {
                var _this__worker__workerPool;
                for (const worker of ((_this__worker__workerPool = this._worker._workerPool) == null ? void 0 : _this__worker__workerPool._workers) || []){
                    var _worker__child;
                    (_worker__child = worker._child) == null ? void 0 : _worker__child.on("exit", (code, signal)=>{
                        if ((code || signal && signal !== "SIGINT") && this._worker) {
                            logger.error(`Next.js build worker exited with code: ${code} and signal: ${signal}`);
                            // We're restarting the worker, so we don't want to exit the parent process
                            if (!this._restarting) {
                                // if a child process doesn't exit gracefully, we want to bubble up the exit code to the parent process
                                process.exit(code ?? 1);
                            }
                        }
                    });
                }
            }
            this._worker.getStdout().pipe(process.stdout);
            this._worker.getStderr().pipe(process.stderr);
        };
        createWorker();
        const onHanging = ()=>{
            const worker = this._worker;
            if (!worker) return;
            const resolve = resolveRestartPromise;
            logger.warn(`Sending SIGTERM signal to Next.js build worker due to timeout${timeout ? ` of ${timeout / 1000} seconds` : ""}. Subsequent errors may be a result of the worker exiting.`);
            this._restarting = true;
            worker.end().then(()=>{
                resolve(RESTARTED);
                createWorker();
            });
        };
        let hangingTimer = false;
        const onActivity = ()=>{
            if (hangingTimer) clearTimeout(hangingTimer);
            hangingTimer = activeTasks > 0 && setTimeout(onHanging, timeout);
        };
        for (const method of farmOptions.exposedMethods){
            if (method.startsWith("_")) continue;
            this[method] = timeout ? async (...args)=>{
                activeTasks++;
                try {
                    let attempts = 0;
                    for(;;){
                        onActivity();
                        const result = await Promise.race([
                            this._worker[method](...args),
                            restartPromise
                        ]);
                        if (result !== RESTARTED) return result;
                        if (onRestart) onRestart(method, args, ++attempts);
                    }
                } finally{
                    activeTasks--;
                    onActivity();
                }
            } : this._worker[method].bind(this._worker);
        }
    }
    end() {
        const worker = this._worker;
        if (!worker) {
            throw new Error("Farm is ended, no more calls can be done to it");
        }
        cleanupWorkers(worker);
        this._worker = undefined;
        return worker.end();
    }
    /**
   * Quietly end the worker if it exists
   */ close() {
        if (this._worker) {
            cleanupWorkers(this._worker);
            this._worker.end();
        }
    }
}

//# sourceMappingURL=worker.js.map