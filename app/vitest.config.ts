import { defineConfig } from "vitest/config";

// Node-environment unit tests over the pure logic (store reducers, wire). Component/DOM tests can add
// happy-dom later. Kept separate from the production tsc (tests are excluded in tsconfig).
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
