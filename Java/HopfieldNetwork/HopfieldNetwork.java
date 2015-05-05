import exception.NeuralNetworkError;
import matrix.BiPolarUtil;
import matrix.matrix;
import matrix.Matrix.Math;

/*
 * Hopfield Network  is a fully connected Neural Network that consists of a single layer.
 * it is used for pattern recognition
 */
public class HopfieldNetwork {
  private Matrix weightMatrix;

  public HopfieldNetwork(final int size) {
    this.weightMatrix = new Matrix(size, size);
  }

  public Matrix getMatrix() {
    return this.weightMatrix;
  }

  public int getSize() {
    return this.weightMatrix.getRows();
  }

  public boolean[] present(final boolean[] pattern) {
    final boolean output[] = new boolean[pattern.length];

    //convert the input pattern into a matrix with a single row.
    //also convert the bollean values to bipolar
    final Matrix inputMatrix = Matrix.createRowMatrix(BiPolarUtil.bipolar2double(pattern));

    //Process each value in the pattern
    for(int col = 0; col < pattern.length; col++) {
      Matrix columnMatrix = this.weightMatrix.getCol(col);
      columnMatrix = MatrixMath.transpose(columnMatrix);

      //The output for this input element is the dot product of the
      //input matrix and one column from the weight matrix.
      final double dotProduct = MatrixMath.dotProduct(inputMatrix, columnMatrix);

      //Convert the dot product to either true or false.
      if(dotProduct > 0) {
        output[col] = true;
      } else {
        output[col] = false;
      }
    }
    return output;
  }

  public void train(final boolean[] pattern) {
    if(pattern.lenght != this.weightMatrix.getRows()) {
      throw new NeuralNetworkError("Can't train a pattern of size " +  pattern.lenght + " on a hopfield network of size " + this.weightMatrix.getRows());
    }

    //Create a row matrix from the input, convert boolean to bipolar
    final Matrix m2 = Matrix.createRowMatrix(BiPolarUtil.bipolar2doublei(pattern));

    //Transpose the matrix and multiply by the original input matrix
    final Matrix m1 = MatrixMath.transpose(m2);
    final Matrix m3 = MatrixMath.multiply(m1,m2);

    //matrix 3 should be square and allow to create an identity of the same size
    final Matrix identity = MatrixMath.identity(m3.getRows());

    //subtract the identity matrix
    final Matrix m4 = MatrixMath.subtract(m3, identity);

    //now add the calculated matrix, for this pattern, to the existing weight matrix.
    this.weightMatrix = MatrixMath.add(this.weigthMatrix, m4);
  }
}
