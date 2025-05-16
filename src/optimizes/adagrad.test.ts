import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { Adagrad } from "./adagrad.ts";

describe("Adagrad Optimizer", () => {
	it("should initialize with default parameters", () => {
		const adagrad = new Adagrad();
		assert.strictEqual(adagrad.learningRate, 0.01);
		// @ts-expect-error
		assert.strictEqual(adagrad.epsilon, 1e-8);
	});

	it("should update weights correctly", () => {
		const adagrad = new Adagrad(0.01, 1e-8);
		const weights = new Map<string, number>([
			["w1", 0.5],
			["w2", -0.5],
		]);
		const gradients = new Map<string, number>([
			["w1", 0.1],
			["w2", -0.2],
		]);

		const updatedWeights = adagrad.update(weights, gradients);

		assert.notStrictEqual(updatedWeights.get("w1"), weights.get("w1"));
		assert.notStrictEqual(updatedWeights.get("w2"), weights.get("w2"));
	});
});
