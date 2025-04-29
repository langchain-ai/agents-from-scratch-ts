"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RouterSchema = void 0;
var zod_1 = require("zod");
/**
 * Router schema for analyzing unread emails and routing based on content
 */
exports.RouterSchema = zod_1.z.object({
    reasoning: zod_1.z.string().describe("Step-by-step reasoning behind the classification"),
    classification: zod_1.z.enum(["ignore", "respond", "notify"]).describe("The classification of an email: 'ignore' for irrelevant emails, " +
        "'notify' for important information that doesn't need a response, " +
        "'respond' for emails that need a reply"),
});
