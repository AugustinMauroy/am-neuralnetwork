import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { Convolution } from "./convolution.ts";

describe("Convolution Layer", { concurrency: true }, () => {
	it("should perform a forward pass correctly", () => {
		const conv = new Convolution(3, 1, 1, 0);
		const input = [
			[
				[
					[1, 2, 3],
					[4, 5, 6],
					[7, 8, 9],
				],
			],
		];
		const output = conv.forward(input);
		assert.strictEqual(output.length, 1, "Batch size should remain 1.");
		assert.strictEqual(output[0].length, 1, "Filters should be 1.");
		assert.deepStrictEqual(
			output[0][0].length,
			1,
			"Output height should be 1.",
		);
		assert.deepStrictEqual(
			output[0][0][0].length,
			1,
			"Output width should be 1.",
		);
	});

	it("should perform backward pass and compute input gradients", () => {
		const conv = new Convolution(3, 1, 1, 0);
		const input = [
			[
				[
					[1, 2, 3],
					[4, 5, 6],
					[7, 8, 9],
				],
			],
		];
		conv.forward(input);
		const gradInput = conv.backward([
			[
				[
					[1], // Single value gradient
				],
			],
		]);
		assert.strictEqual(
			gradInput[0][0].length,
			3,
			"Input gradient height should match original input height.",
		);
		assert.strictEqual(
			gradInput[0][0][0].length,
			3,
			"Input gradient width should match original input width.",
		);
	});

	it("should return the same gradient if forward is not called", () => {
		const conv = new Convolution();
		const gradient = [[[[0]]]];
		const result = conv.backward(gradient);
		assert.deepStrictEqual(
			result,
			gradient,
			"Gradient should remain unchanged if forward not called.",
		);
	});

	it("should return the correct output shape", () => {
		const conv = new Convolution(3, 2, 2, 1);
		const shape = conv.getOutputShape([1, 1, 5, 5]);
		assert.deepStrictEqual(shape, [1, 2, 3, 3], "Output shape is incorrect.");
	});

	it("should return the correct input shape after forward", () => {
		const conv = new Convolution(3, 1, 1, 0);
		const input = [[[[1, 2, 3]]]];
		conv.forward(input);
		const shape = conv.getInputShape();
		assert.deepStrictEqual(
			shape,
			[1, 1, 1, 3],
			"Input shape should be stored from last forward call.",
		);
	});

	it("should return the layer name and config", () => {
		const conv = new Convolution(3, 1, 2, 1);
		assert.strictEqual(
			conv.getName(),
			"Convolution",
			"Layer name should match.",
		);
		assert.deepStrictEqual(
			conv.getConfig(),
			{
				kernelSize: 3,
				filters: 1,
				stride: 2,
				padding: 1,
			},
			"Config should match constructor parameters.",
		);
	});

	it("should return a default input shape if forward has not been called", () => {
		const conv = new Convolution();
		const shape = conv.getInputShape();
		assert.deepStrictEqual(
			shape,
			[0, 0, 0, 0],
			"Should return [0,0,0,0] when forward not called.",
		);
	});
});
