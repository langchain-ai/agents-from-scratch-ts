/** @type {import('ts-jest').JestConfigWithTsJest} **/
export default {
  testEnvironment: "node",
  testMatch: ["**/tests/**/*.test.ts"],
  extensionsToTreatAsEsm: [".ts"],
  transform: {
    "^.+\\.tsx?$": [
      "ts-jest",
      {
        tsconfig: "tsconfig.json",
        useESM: true,
      },
    ],
  },
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/$1",
    "^(\\.{1,2}/.*)\\.js$": "$1",
  },

  transformIgnorePatterns: ["node_modules/(?!(@langchain|langchain|@jest)/)"],
  setupFilesAfterEnv: ["<rootDir>/tests/setup.mjs"],
  testTimeout: 30000, // For LLM calls
  moduleDirectories: ["node_modules", "src"],
};
