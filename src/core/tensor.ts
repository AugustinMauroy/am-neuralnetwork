class Tensor {
  private data: number[][];
  private shape: number[];

  constructor(data: number[][]) {
    this.data = data;
    this.shape = [data.length, data[0].length];
  }

  public add(other: Tensor): Tensor {
    if (this.shape[0] !== other.shape[0] || this.shape[1] !== other.shape[1]) {
      throw new Error("Shapes of tensors must match for addition.");
    }

    const resultData = this.data.map((row, i) =>
      row.map((value, j) => value + other.data[i][j])
    );

    return new Tensor(resultData);
  }

  public multiply(other: Tensor): Tensor {
    if (this.shape[1] !== other.shape[0]) {
      throw new Error("Inner dimensions must match for multiplication.");
    }

    const resultData = Array.from(
      { length: this.shape[0] },
      () => Array(other.shape[1]).fill(0),
    );

    for (let i = 0; i < this.shape[0]; i++) {
      for (let j = 0; j < other.shape[1]; j++) {
        for (let k = 0; k < this.shape[1]; k++) {
          resultData[i][j] += this.data[i][k] * other.data[k][j];
        }
      }
    }

    return new Tensor(resultData);
  }

  public reshape(newShape: number[]): Tensor {
    const totalElements = this.shape.reduce((a, b) => a * b, 1);
    const newTotalElements = newShape.reduce((a, b) => a * b, 1);

    if (totalElements !== newTotalElements) {
      throw new Error("Total number of elements must remain the same.");
    }

    const flatData = this.data.flat();
    const resultData: number[][] = [];
    let index = 0;

    for (let i = 0; i < newShape[0]; i++) {
      const row: number[] = [];
      for (let j = 0; j < newShape[1]; j++) {
        row.push(flatData[index++]);
      }
      resultData.push(row);
    }

    return new Tensor(resultData);
  }

  public getShape(): number[] {
    return this.shape;
  }
}

export default Tensor;
