import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { SGD } from "./sgd.ts";

describe("SGD Optimizer", () => {
	it("should initialize with default parameters", () => {
		const sgd = new SGD();

		assert.strictEqual(sgd.learningRate, 0.01);
	});

	it("should update weights correctly", () => {
		const sgd = new SGD(0.01);
		const weights = new Map<string, number>([
			["w1", 0.5],
			["w2", -0.5],
		]);
		const gradients = new Map<string, number>([
			["w1", 0.1],
			["w2", -0.2],
		]);

		const updatedWeights = sgd.update(weights, gradients);

		assert.notStrictEqual(updatedWeights.get("w1"), weights.get("w1"));
		assert.notStrictEqual(updatedWeights.get("w2"), weights.get("w2"));
	});

	it("getName should return 'SGD'", () => {
		const sgd = new SGD();

		assert.strictEqual(sgd.getName(), "SGD");
	});

	it("getConfig should return the correct configuration", () => {
		const sgd = new SGD(0.01);
		const config = sgd.getConfig();

		assert.deepStrictEqual(config, {
			name: "SGD",
			learningRate: 0.01,
		});
	});
});
