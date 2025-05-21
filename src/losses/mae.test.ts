import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { MeanAbsoluteError } from "./mae.ts";

describe("MeanAbsoluteError", () => {
	it("should return 0 when predictions are perfect", () => {
		const yTrue = [1, 2, 3];
		const yPred = [1, 2, 3];
		const mae = new MeanAbsoluteError();
		const result = mae.calculate(yTrue, yPred);
		assert.strictEqual(result, 0);
	});

	it("should calculate mean absolute error correctly", () => {
		const yTrue = [1, 2, 3];
		const yPred = [2, 3, 4];
		const mae = new MeanAbsoluteError();
		const result = mae.calculate(yTrue, yPred); // All errors = 1, mean = 1
		assert.strictEqual(result, 1);
	});

	it("should handle floating point differences", () => {
		const yTrue = [1, 2, 3];
		const yPred = [1.1, 1.9, 3.2];
		const mae = new MeanAbsoluteError();
		const result = mae.calculate(yTrue, yPred);
		assert.ok(Math.abs(result - 0.13333333333333344) < 1e-10);
	});

	it("should throw an error for different length arrays", () => {
		const yTrue = [1, 2];
		const yPred = [1, 2, 3];
		const mae = new MeanAbsoluteError();
		assert.throws(
			() => {
				mae.calculate(yTrue, yPred);
			},
			{
				message: "Predictions and targets must have the same length.",
			},
		);
	});

	it("should return 0 for empty arrays", () => {
		const mae = new MeanAbsoluteError();
		const result = mae.calculate([], []);
		assert.strictEqual(result, 0);
	});
});
