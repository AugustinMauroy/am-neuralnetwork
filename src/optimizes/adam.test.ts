import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { Adam } from "./adam.ts";

describe("Adam Optimizer", () => {
	it("should initialize with default parameters", () => {
		const adam = new Adam();
		assert.strictEqual(adam.learningRate, 0.001);
		// @ts-expect-error
		assert.strictEqual(adam.beta1, 0.9);
		// @ts-expect-error
		assert.strictEqual(adam.beta2, 0.999);
		// @ts-expect-error
		assert.strictEqual(adam.epsilon, 1e-8);
	});

	it("should update weights correctly", () => {
		const adam = new Adam(0.01, 0.9, 0.999, 1e-8);
		const weights = new Map<string, number>([
			["w1", 0.5],
			["w2", -0.5],
		]);
		const gradients = new Map<string, number>([
			["w1", 0.1],
			["w2", -0.2],
		]);

		const updatedWeights = adam.update(weights, gradients);

		assert.notStrictEqual(updatedWeights.get("w1"), weights.get("w1"));
		assert.notStrictEqual(updatedWeights.get("w2"), weights.get("w2"));
	});
});
