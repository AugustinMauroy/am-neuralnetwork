import type { Layer } from "../core/model.ts";

export abstract class Activation implements Layer {
  abstract forward(input: number[][]): number[][];
  abstract backward(outputGradient: number[][]): number[][];
}

export class Sigmoid extends Activation {
  private lastInput!: number[][];

  forward(input: number[][]): number[][] {
    this.lastInput = input;
    return input.map((row) => row.map((x) => 1 / (1 + Math.exp(-x))));
  }

  backward(outputGradient: number[][]): number[][] {
    if (!this.lastInput) {
      throw new Error(
        "Forward pass must be called before backward pass for Sigmoid.",
      );
    }
    const sigmoidOutput = this.lastInput.map((row) =>
      row.map((x) => 1 / (1 + Math.exp(-x)))
    );
    return outputGradient.map((gradRow, i) =>
      gradRow.map((grad, j) => {
        const s = sigmoidOutput[i][j];
        return grad * s * (1 - s);
      })
    );
  }
}

export class ReLU extends Activation {
  private lastInput!: number[][];

  forward(input: number[][]): number[][] {
    this.lastInput = input;
    return input.map((row) => row.map((x) => Math.max(0, x)));
  }

  backward(outputGradient: number[][]): number[][] {
    if (!this.lastInput) {
      throw new Error(
        "Forward pass must be called before backward pass for ReLU.",
      );
    }
    return outputGradient.map((gradRow, i) =>
      gradRow.map((grad, j) => (this.lastInput[i][j] > 0 ? grad : 0))
    );
  }
}

export class Tanh extends Activation {
  private lastInput!: number[][];

  forward(input: number[][]): number[][] {
    this.lastInput = input;
    return input.map((row) => row.map((x) => Math.tanh(x)));
  }

  backward(outputGradient: number[][]): number[][] {
    if (!this.lastInput) {
      throw new Error(
        "Forward pass must be called before backward pass for Tanh.",
      );
    }
    return outputGradient.map((gradRow, i) =>
      gradRow.map((grad, j) => {
        const tanhX = Math.tanh(this.lastInput[i][j]);
        return grad * (1 - tanhX * tanhX);
      })
    );
  }
}

export class Softmax extends Activation {
  private lastOutput!: number[][]; // To store the output of the forward pass

  forward(input: number[][]): number[][] {
    this.lastOutput = input.map((row) => {
      // Subtract max for numerical stability
      const maxVal = Math.max(...row);
      const exps = row.map((x) => Math.exp(x - maxVal));
      const sumExps = exps.reduce((sum, val) => sum + val, 0);
      return exps.map((exp) => exp / sumExps);
    });
    return this.lastOutput;
  }

  backward(outputGradient: number[][]): number[][] {
    if (!this.lastOutput) {
      throw new Error(
        "Forward pass must be called before backward pass for Softmax.",
      );
    }
    if (outputGradient.length !== this.lastOutput.length) {
      throw new Error(
        "Output gradient and last output must have the same batch size.",
      );
    }

    return outputGradient.map((gradRow, i) => {
      const softmaxOutputRow = this.lastOutput[i];
      if (gradRow.length !== softmaxOutputRow.length) {
        throw new Error(
          "Gradient row and softmax output row must have the same number of units.",
        );
      }

      // Calculate the dot product of the output gradient and the softmax output for this sample
      const dotProduct = gradRow.reduce(
        (sum, gradVal, j) => sum + gradVal * softmaxOutputRow[j],
        0,
      );

      // Calculate the input gradient for this sample
      return softmaxOutputRow.map(
        (sVal, j) => sVal * (gradRow[j] - dotProduct),
      );
    });
  }
}
