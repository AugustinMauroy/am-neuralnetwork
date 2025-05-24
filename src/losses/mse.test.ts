import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { MeanSquaredError } from "./mse.ts";

describe("MeanSquaredError", () => {
	it("should calculate the mean squared error correctly", () => {
		const yTrue = [1, 2, 3];
		const yPred = [1, 2, 3];
		const mse = new MeanSquaredError();

		const result = mse.calculate(yTrue, yPred);

		assert.strictEqual(result, 0);
	});

	it("should calculate the mean squared error with different values", () => {
		const yTrue = [1, 2, 3];
		const yPred = [4, 5, 6];
		const mse = new MeanSquaredError();

		const result = mse.calculate(yTrue, yPred);

		assert.strictEqual(result, 9);
	});

	it("should throw an error for different length arrays", () => {
		const yTrue = [1, 2, 3];
		const yPred = [1, 2];

		const mse = new MeanSquaredError();

		assert.throws(
			() => {
				mse.calculate(yTrue, yPred);
			},
			{
				message: "Predictions and targets must have the same length.",
			},
		);
	});

	it("should return 0 for empty arrays", () => {
		const yTrue: number[] = [];
		const yPred: number[] = [];
		const mse = new MeanSquaredError();

		const result = mse.calculate(yTrue, yPred);
		
		assert.strictEqual(result, 0);
	});

	it("name should be accessible", () => {
		const mse = new MeanSquaredError();
		assert.strictEqual(mse.name, "MeanSquaredError");
	});
});
