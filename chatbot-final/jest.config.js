/** @type {import('ts-jest').JestConfigWithTsJest} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1', // adjust if your alias is @ pointing somewhere else
  },
  setupFilesAfterEnv: ['@testing-library/jest-dom/extend-expect'],
};