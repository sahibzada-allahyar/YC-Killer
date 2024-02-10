"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.reservationsUrl = exports.workersUrl = exports.activitiesUrl = exports.tasksUrl = exports.taskQueuesUrl = exports.workspacesUrl = exports.defaultEventBridgePolicies = exports.defaultWorkerPolicies = void 0;
const TaskRouterCapability_1 = require("./TaskRouterCapability");
const EVENT_URL_BASE = "https://event-bridge.twilio.com/v1/wschannels";
const TASKROUTER_BASE_URL = "https://taskrouter.twilio.com";
const TASKROUTER_VERSION = "v1";
/**
 * Build the default Policies for a worker
 *
 * @param version - TaskRouter version
 * @param workspaceSid - workspace sid
 * @param workerSid - worker sid
 * @returns list of Policies
 */
function defaultWorkerPolicies(version, workspaceSid, workerSid) {
    var activities = new TaskRouterCapability_1.Policy({
        url: [
            TASKROUTER_BASE_URL,
            version,
            "Workspaces",
            workspaceSid,
            "Activities",
        ].join("/"),
        method: "GET",
        allow: true,
    });
    var tasks = new TaskRouterCapability_1.Policy({
        url: [
            TASKROUTER_BASE_URL,
            version,
            "Workspaces",
            workspaceSid,
            "Tasks",
            "**",
        ].join("/"),
        method: "GET",
        allow: true,
    });
    var reservations = new TaskRouterCapability_1.Policy({
        url: [
            TASKROUTER_BASE_URL,
            version,
            "Workspaces",
            workspaceSid,
            "Workers",
            workerSid,
            "Reservations",
            "**",
        ].join("/"),
        method: "GET",
        allow: true,
    });
    var workerFetch = new TaskRouterCapability_1.Policy({
        url: [
            TASKROUTER_BASE_URL,
            version,
            "Workspaces",
            workspaceSid,
            "Workers",
            workerSid,
        ].join("/"),
        method: "GET",
        allow: true,
    });
    return [activities, tasks, reservations, workerFetch];
}
exports.defaultWorkerPolicies = defaultWorkerPolicies;
/**
 * Build the default Event Bridge Policies
 *
 * @param accountSid - account sid
 * @param channelId - channel id
 * @returns list of Policies
 */
function defaultEventBridgePolicies(accountSid, channelId) {
    var url = [EVENT_URL_BASE, accountSid, channelId].join("/");
    return [
        new TaskRouterCapability_1.Policy({
            url: url,
            method: "GET",
            allow: true,
        }),
        new TaskRouterCapability_1.Policy({
            url: url,
            method: "POST",
            allow: true,
        }),
    ];
}
exports.defaultEventBridgePolicies = defaultEventBridgePolicies;
/**
 * Generate TaskRouter workspace url
 *
 * @param workspaceSid - workspace sid or '**' for all workspaces
 * @returns generated url
 */
function workspacesUrl(workspaceSid) {
    return [TASKROUTER_BASE_URL, TASKROUTER_VERSION, "Workspaces", workspaceSid]
        .filter((item) => typeof item === "string")
        .join("/");
}
exports.workspacesUrl = workspacesUrl;
/**
 * Generate TaskRouter task queue url
 *
 * @param workspaceSid - workspace sid
 * @param taskQueueSid - task queue sid or '**' for all task queues
 * @returns generated url
 */
function taskQueuesUrl(workspaceSid, taskQueueSid) {
    return [workspacesUrl(workspaceSid), "TaskQueues", taskQueueSid]
        .filter((item) => typeof item === "string")
        .join("/");
}
exports.taskQueuesUrl = taskQueuesUrl;
/**
 * Generate TaskRouter task url
 *
 * @param workspaceSid - workspace sid
 * @param taskSid - task sid or '**' for all tasks
 * @returns generated url
 */
function tasksUrl(workspaceSid, taskSid) {
    return [workspacesUrl(workspaceSid), "Tasks", taskSid]
        .filter((item) => typeof item === "string")
        .join("/");
}
exports.tasksUrl = tasksUrl;
/**
 * Generate TaskRouter activity url
 *
 * @param workspaceSid - workspace sid
 * @param activitySid - activity sid or '**' for all activities
 * @returns generated url
 */
function activitiesUrl(workspaceSid, activitySid) {
    return [workspacesUrl(workspaceSid), "Activities", activitySid]
        .filter((item) => typeof item === "string")
        .join("/");
}
exports.activitiesUrl = activitiesUrl;
/**
 * Generate TaskRouter worker url
 *
 * @param workspaceSid - workspace sid
 * @param workerSid - worker sid or '**' for all workers
 * @returns generated url
 */
function workersUrl(workspaceSid, workerSid) {
    return [workspacesUrl(workspaceSid), "Workers", workerSid]
        .filter((item) => typeof item === "string")
        .join("/");
}
exports.workersUrl = workersUrl;
/**
 * Generate TaskRouter worker reservation url
 *
 * @param workspaceSid - workspace sid
 * @param workerSid - worker sid
 * @param reservationSid - reservation sid or '**' for all reservations
 * @returns generated url
 */
function reservationsUrl(workspaceSid, workerSid, reservationSid) {
    return [workersUrl(workspaceSid, workerSid), "Reservations", reservationSid]
        .filter((item) => typeof item === "string")
        .join("/");
}
exports.reservationsUrl = reservationsUrl;
