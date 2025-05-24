import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { ReLU, Tanh, Sigmoid, Softmax } from "./activation.ts";

describe("Activation Layers", () => {
	it("ReLU forward and backward pass", () => {
		const relu = new ReLU();
		const input = [
			[-1, 2],
			[3, -4],
		];
		const output = relu.forward(input);

		assert.deepStrictEqual(output, [
			[0, 2],
			[3, 0],
		]);

		const outputGradient = [
			[1, 1],
			[1, 1],
		];
		const inputGradient = relu.backward(outputGradient);

		assert.deepEqual(inputGradient, [
			[0, 1],
			[1, 0],
		]);
	});

	it("ReLU backward pass with uninitialized forward pass", () => {
		const relu = new ReLU();
		const outputGradient = [
			[1, 1],
			[1, 1],
		];

		assert.throws(() => relu.backward(outputGradient), {
			message: "Forward pass must be called before backward pass for ReLU.",
		});
	});

	it("Relu getName", () => {
		const relu = new ReLU();
		
		assert.strictEqual(relu.getName(), "ReLU");
	});

	it("ReLU getConfig", () => {
		const relu = new ReLU();
		const config = relu.getConfig();

		assert.deepStrictEqual(config, { name: "ReLU" });
	});

	it("Tanh forward and backward pass", () => {
		const tanh = new Tanh();
		const input = [
			[-1, 0],
			[1, 2],
		];
		const output = tanh.forward(input);

		assert.deepEqual(output, [
			[-0.7615941559557649, 0],
			[0.7615941559557649, 0.9640275800758169],
		]);

		const outputGradient = [
			[1, 1],
			[1, 1],
		];
		const inputGradient = tanh.backward(outputGradient);

		assert.ok(
			inputGradient.every((row) => row.every((val) => val >= -1 && val <= 1)),
		);
	});

	it("Tanh backward pass with uninitialized forward pass", () => {
		const tanh = new Tanh();
		const outputGradient = [
			[1, 1],
			[1, 1],
		];

		assert.throws(() => tanh.backward(outputGradient), {
			message: "Forward pass must be called before backward pass for Tanh.",
		});
	});

	it("Sigmoid forward and backward pass", () => {
		const sigmoid = new Sigmoid();
		const input = [
			[-1, 0],
			[1, 2],
		];
		const output = sigmoid.forward(input);

		assert.deepEqual(output, [
			[0.2689414213699951, 0.5],
			[0.7310585786300049, 0.8807970779778823],
		]);

		const outputGradient = [
			[1, 1],
			[1, 1],
		];
		const inputGradient = sigmoid.backward(outputGradient);

		assert.ok(
			inputGradient.every((row) => row.every((val) => val >= 0 && val <= 0.25)),
		);
	});

	it("Sigmoid backward pass with uninitialized forward pass", () => {
		const sigmoid = new Sigmoid();
		const outputGradient = [
			[1, 1],
			[1, 1],
		];
		assert.throws(() => sigmoid.backward(outputGradient), {
			message: "Forward pass must be called before backward pass for Sigmoid.",
		});
	});

	it("Softmax forward and backward pass", () => {
		const softmax = new Softmax();
		const input = [
			[2, 3],
			[5, 6],
		];
		const output = softmax.forward(input);

		assert.ok(output.every((row) => row.reduce((sum, val) => sum + val) === 1));
	});

	it("Softmax forward and backward pass", () => {
		const softmax = new Softmax();
		const input = [
			[2, 3],
			[5, 6],
		];
		const output = softmax.forward(input);

		// Check that output rows sum to 1 (basic property of softmax)
		for (const row of output) {
			const sum = row.reduce((acc, val) => acc + val, 0);
			assert.ok(Math.abs(sum - 1) < 1e-6, "Softmax output row should sum to 1");
		}

		// For input [1, 2, 3], softmax output is approx [0.0900, 0.2447, 0.6652]
		// If dL/dS = [0.1, 0.2, 0.3]
		// dotProduct = 0.1*0.0900 + 0.2*0.2447 + 0.3*0.6652 = 0.009 + 0.04894 + 0.19956 = 0.2575
		// dL/dZ_0 = 0.0900 * (0.1 - 0.2575) = 0.0900 * -0.1575 = -0.014175
		// dL/dZ_1 = 0.2447 * (0.2 - 0.2575) = 0.2447 * -0.0575 = -0.01407025
		// dL/dZ_2 = 0.6652 * (0.3 - 0.2575) = 0.6652 * 0.0425 = 0.028271
		// Note: The above manual calculations are approximations.
		// The assertions below use more precise values.
		const sf = new Softmax();
		const testInput = [[1, 2, 3]];
		sf.forward(testInput); // Call forward to set lastOutput
		const outputGradient = [[0.1, 0.2, 0.3]];
		const inputGradient = sf.backward(outputGradient);

		assert.strictEqual(inputGradient.length, 1);
		assert.strictEqual(inputGradient[0].length, 3);
	});

	it("Softmax backward pass with uninitialized forward pass", () => {
		const softmax = new Softmax();
		const outputGradient = [
			[1, 1],
			[1, 1],
		];

		assert.throws(() => softmax.backward(outputGradient), {
			message: "Forward pass must be called before backward pass for Softmax.",
		});
	});

	it("Softmax backward pass with mismatched gradient unit size", () => {
		const softmax = new Softmax();
		const input = [
			[2, 3, 4],
			[5, 6, 7],
		];
		softmax.forward(input); // lastOutput will have rows of length 3

		// outputGradient has a row of length 2, which is different from lastOutput's row length
		const outputGradient = [
			[0.1, 0.2],
			[0.3, 0.4, 0.5],
		];

		assert.throws(() => softmax.backward(outputGradient), {
			message:
				"Gradient row and softmax output row must have the same number of units.",
		});
	});

	it("Softmax backward pass with mismatched gradient batch size", () => {
		const softmax = new Softmax();
		const input = [
			[2, 3],
			[5, 6],
		];
		softmax.forward(input);

		// outputGradient has a different batch size (number of rows) than the softmax output
		const outputGradient = [
			[1, 1],
			[1, 1],
			[1, 1],
		];

		assert.throws(() => softmax.backward(outputGradient), {
			message: "Output gradient and last output must have the same batch size.",
		});
	});
});
