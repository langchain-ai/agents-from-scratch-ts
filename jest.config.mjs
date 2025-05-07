/** @type {import('jest').Config} */
export default {
  preset: 'ts-jest/presets/js-with-ts-esm',
  testEnvironment: 'node',
  testMatch: ['**/tests-ts/**/*.test.ts'],
  extensionsToTreatAsEsm: ['.ts'],
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      tsconfig: 'tsconfig.json',
      useESM: true,
    }],
  },
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/$1',
    '^(\\.{1,2}/.*)\\.js$': '$1',
  },
  transformIgnorePatterns: [
    'node_modules/(?!(@langchain|langchain|@jest)/)'
  ],
  rootDir: '../',
  setupFilesAfterEnv: ['<rootDir>/tests-ts/setup.mjs'],
  testTimeout: 30000, // For LLM calls
  moduleDirectories: ['node_modules', 'src'],
  resolver: 'jest-ts-webcompat-resolver',
}; 