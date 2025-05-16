import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { Tensor } from "./tensor.ts";

describe("Tensor", () => {
	describe("constructor", () => {
		it("should create a tensor with valid 2D array data", () => {
			const data = [
				[1, 2],
				[3, 4],
			];
			const tensor = new Tensor(data);

			assert.deepStrictEqual(tensor.getData(), data);
			assert.deepStrictEqual(tensor.getShape(), [2, 2]);
		});

		it("should throw an error for non-array input", () => {
			assert.throws(
				() => new Tensor(null),
				/Input data must be a non-empty 2D array of numbers./,
			);
		});

		it("should throw an error for empty array input", () => {
			assert.throws(
				() => new Tensor([]),
				/Input data must be a non-empty 2D array of numbers./,
			);
		});

		it("should throw an error for array of non-arrays", () => {
			assert.throws(
				// @ts-expect-error
				() => new Tensor([1, 2]),
				/Input data must be a non-empty 2D array of numbers./,
			);
		});
	});

	describe("add", () => {
		it("should add two tensors of the same shape", () => {
			const tensorA = new Tensor([
				[1, 2],
				[3, 4],
			]);
			const tensorB = new Tensor([
				[5, 6],
				[7, 8],
			]);

			const result = tensorA.add(tensorB);

			assert.deepStrictEqual(result.getData(), [
				[6, 8],
				[10, 12],
			]);
		});

		it("should throw an error when adding tensors of different shapes", () => {
			const tensorA = new Tensor([
				[1, 2],
				[3, 4],
			]);
			const tensorB = new Tensor([[1, 2, 3]]);

			assert.throws(
				() => tensorA.add(tensorB),
				/Shapes of tensors must match for addition./,
			);
		});
	});

	describe("multiply", () => {
		it("should multiply two compatible tensors", () => {
			const tensorA = new Tensor([
				[1, 2],
				[3, 4],
			]); // 2x2
			const tensorB = new Tensor([
				[5, 6],
				[7, 8],
			]); // 2x2

			const result = tensorA.multiply(tensorB);

			assert.deepStrictEqual(result.getData(), [
				[19, 22],
				[43, 50],
			]);
		});

		it("should multiply a 2x3 tensor by a 3x2 tensor", () => {
			const tensorC = new Tensor([
				[1, 2, 3],
				[4, 5, 6],
			]); // 2x3
			const tensorD = new Tensor([
				[7, 8],
				[9, 10],
				[11, 12],
			]); // 3x2

			const productCD = tensorC.multiply(tensorD);

			assert.deepStrictEqual(productCD.getData(), [
				[58, 64],
				[139, 154],
			]);
		});

		it("should throw an error for incompatible tensor multiplication", () => {
			const tensorA = new Tensor([
				[1, 2],
				[3, 4],
			]); // 2x2
			const tensorB = new Tensor([[1, 2, 3]]); // 1x3

			assert.throws(
				() => tensorA.multiply(tensorB),
				/Inner dimensions must match for multiplication. Got 2 and 1/,
			);
		});
	});

	describe("reshape", () => {
		it("should reshape a tensor to a valid new shape", () => {
			const tensor = new Tensor([
				[1, 2, 3, 4],
				[5, 6, 7, 8],
			]); // 2x4

			const reshaped = tensor.reshape([4, 2]);

			assert.deepStrictEqual(reshaped.getData(), [
				[1, 2],
				[3, 4],
				[5, 6],
				[7, 8],
			]);

			assert.deepStrictEqual(reshaped.getShape(), [4, 2]);
		});

		it("should throw an error if total number of elements changes", () => {
			const tensor = new Tensor([
				[1, 2],
				[3, 4],
			]);

			assert.throws(
				() => tensor.reshape([3, 2]),
				/Total number of elements must remain the same./,
			);
		});

		it("should throw an error if new shape is not 2D", () => {
			const tensor = new Tensor([
				[1, 2, 3, 4],
				[5, 6, 7, 8],
			]);

			assert.throws(
				() => tensor.reshape([8]),
				/New shape must be a 2D shape \[rows, columns]./,
			);
		});
	});

	describe("getShape", () => {
		it("should return the correct shape", () => {
			const tensor = new Tensor([
				[1, 2, 3],
				[4, 5, 6],
			]);

			assert.deepStrictEqual(tensor.getShape(), [2, 3]);
		});

		it("should return a copy of the shape array", () => {
			const tensor = new Tensor([
				[1, 2],
				[3, 4],
			]);
			const shape = tensor.getShape();

			shape[0] = 5; // Modify the returned shape
			assert.deepStrictEqual(tensor.getShape(), [2, 2]); // Original shape should be unchanged
		});
	});

	describe("getData", () => {
		it("should return the correct data", () => {
			const data = [
				[1, 2],
				[3, 4],
			];
			const tensor = new Tensor(data);

			assert.deepStrictEqual(tensor.getData(), data);
		});

		it("should return a deep copy of the data", () => {
			const data = [
				[1, 2],
				[3, 4],
			];
			const tensor = new Tensor(data);

			const retrievedData = tensor.getData();
			retrievedData[0][0] = 99; // Modify the retrieved data

			assert.deepStrictEqual(tensor.getData(), [
				[1, 2],
				[3, 4],
			]); // Original data should be unchanged
		});
	});
});
