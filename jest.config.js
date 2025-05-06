export default {
  preset: 'ts-jest',
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
  rootDir: './',
  setupFilesAfterEnv: ['<rootDir>/tests-ts/setup.ts'],
  testTimeout: 30000, // For LLM calls
}; 