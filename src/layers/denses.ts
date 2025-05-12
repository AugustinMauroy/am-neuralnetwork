import type { Layer } from "../core/model.ts";

export class Dense implements Layer {
  private inputUnits: number;
  private outputUnits: number;
  private weights: number[][];
  private biases: number[];
  private lastInput!: number[][];

  constructor(inputUnits: number, outputUnits: number) {
    this.inputUnits = inputUnits;
    this.outputUnits = outputUnits;
    this.weights = this.initializeWeights();
    this.biases = this.initializeBiases();
  }

  private initializeWeights(): number[][] {
    const weights = [];
    for (let i = 0; i < this.inputUnits; i++) {
      weights[i] = Array(this.outputUnits).fill(0).map(() => Math.random());
    }
    return weights;
  }

  private initializeBiases(): number[] {
    return Array(this.outputUnits).fill(0).map(() => Math.random());
  }

  public forward(input: number[][]): number[][] {
    this.lastInput = input;
    const output = [];
    for (let i = 0; i < input.length; i++) {
      const row = [];
      for (let j = 0; j < this.outputUnits; j++) {
        const weightedSum = input[i].reduce(
          (sum, value, index) => sum + value * this.weights[index][j],
          this.biases[j],
        );
        row.push(weightedSum);
      }
      output.push(row);
    }
    return output;
  }

  public backward(outputGradient: number[][]): number[][] {
    // outputGradient (dL/dY): shape (batchSize, this.outputUnits)
    // this.weights (W): shape (this.inputUnits, this.outputUnits), W_ij is weight from input i to output j.
    // inputGradient (dL/dX): shape (batchSize, this.inputUnits)
    // dL/dX_i = sum_j (dL/dY_j * W_ij)

    const batchSize = outputGradient.length;
    if (batchSize === 0) {
      return [];
    }
    // Assuming outputGradient[0] exists and has a length property if batchSize > 0
    if (outputGradient[0].length !== this.outputUnits) {
      throw new Error("Output gradient dimension mismatch.");
    }

    const inputGradient: number[][] = Array(batchSize).fill(null).map(() =>
      Array(this.inputUnits).fill(0)
    );

    for (let b = 0; b < batchSize; b++) { // For each sample in the batch
      for (let i = 0; i < this.inputUnits; i++) { // For each input unit of this layer
        let gradSum = 0;
        for (let j = 0; j < this.outputUnits; j++) { // For each output unit of this layer
          gradSum += outputGradient[b][j] * this.weights[i][j];
        }
        inputGradient[b][i] = gradSum;
      }
    }
    return inputGradient;
  }
}
