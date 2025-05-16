import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { RMSprop } from "./rmsprop.ts";

describe("RMSprop Optimizer", () => {
	it("should initialize with default parameters", () => {
		const rmsprop = new RMSprop();
		assert.strictEqual(rmsprop.learningRate, 0.001);
		// @ts-expect-error
		assert.strictEqual(rmsprop.decayRate, 0.9);
		// @ts-expect-error
		assert.strictEqual(rmsprop.epsilon, 1e-8);
	});

	it("should update weights correctly", () => {
		const rmsprop = new RMSprop(0.01, 0.9, 1e-8);
		const weights = new Map<string, number>([
			["w1", 0.5],
			["w2", -0.5],
		]);
		const gradients = new Map<string, number>([
			["w1", 0.1],
			["w2", -0.2],
		]);

		const updatedWeights = rmsprop.update(weights, gradients);

		assert.notStrictEqual(updatedWeights.get("w1"), weights.get("w1"));
		assert.notStrictEqual(updatedWeights.get("w2"), weights.get("w2"));
	});
});
