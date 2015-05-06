using System;
namespace NeuralNetworkFeedForward
{
  class FeedForwardProgram
  {
    static void Main(string[] args)
    {
      try
      {
        Console.WriteLine("\nBegin neural network feed-forward demo\n");

        Console.WriteLine("Creating a 3-input, 4-hidden, 2-output NN");
        Console.WriteLine("Using log-sigmoid function");

        const int numInput = 3;
        const int numHidden = 4;
        const int numOutput = 2;

        NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);

        const int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;

        double[] weights = new double[numWeights] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, -2.0, -6.0, -1.0, -7.0, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, -2.5, -5.0 };

        Console.WriteLine("\nWeights and biases are:");
        ShowVector(weights, 2);

        Console.WriteLine("Loading neural network weights and biases");
        nn.SetWeights(weights);

        Console.WriteLine("\nSetting neural network inputs:");
        double[] xValues = new double[] { 2.0, 3.0, 4.0 };
        ShowVector(xValues, 2);

        Console.WriteLine("Loading inputs and computing outputs\n");
        double[] yValues = nn.ComputeOutputs(xValues);

        Console.WriteLine("\nNeural network outputs are:");
        ShowVector(yValues, 4);

        Console.WriteLine("\nEnd neural network demo\n");
        Console.ReadLine();
      }
      catch (Exception ex)
      {
        Console.WriteLine(ex.Message);
        Console.ReadLine();
      }
    } // Main

    public static void ShowVector(double[] vector, int decimals) 
    {
      for (int i = 0; i < vector.Length; ++i)
      {
        if (i > 0 && i % 12 == 0) // max of 12 values per row 
          Console.WriteLine("");
        if (vector[i] >= 0.0) Console.Write(" ");
        Console.Write(vector[i].ToString("F" + decimals) + " "); // 2 decimals
      }
      Console.WriteLine("\n");
    }

    public static void ShowMatrix(double[double[]] matrix, int numRows) 
    {
      int ct = 0;
      if (numRows == -1) numRows = int.MaxValue; // if numRows == -1, show all rows
      for (int i = 0; i < matrix.Length && ct < numRows; ++i)
      {
        for (int j = 0; j < matrix[0].Length; ++j)
        {
          if (matrix[i][j] >= 0.0) Console.Write(" ");
          Console.Write(matrix[i][j].ToString("F2") + " ");
        }
        Console.WriteLine("");
        ++ct;
      }
      Console.WriteLine("");
    }
  } // Program

  public class NeuralNetwork
  {
    private int[] layerSizes[];

    private double[] inputs;
    private double[double[]] weights;
    private double[double[]] biases;
    private double[] outputs;

    public NeuralNetwork(int[] layerSizes)
    {
      this.layerSizes = layerSizes;
      
      MakeMatrix(layerSizes);
    }

    private static double MakeMatrix(int[] layerSizes) 
    {
      inputs = new double[layerSizes[0]];
      outputs = new double[layerSizes[layerSizes.Length-1]];
      for (int i = 1; i < layerSizes.Length - 1; i++)
      {
        biases[i-1] = new double[layerSizes[i]];
        weights[i] = new double[layerSizes[i]*layerSizes[i+1]];
      }
      return result;
    }

    public void SetWeights(double[] weights) 
    {
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      if (weights.Length != numWeights)
        throw new Exception("Bad weights array");

      int k = 0; // Points into weights param

      for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
          ihWeights[i][j] = weights[k++];

      for (int i = 0; i < numHidden; ++i)
        ihBiases[i] = weights[k++];

      for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
          hoWeights[i][j] = weights[k++];

      for (int i = 0; i < numOutput; ++i)
        hoBiases[i] = weights[k++];
    }

    public double[] ComputeOutputs(double[] xValues) 
    {
      if (xValues.Length != numInput)
        throw new Exception("Bad inputs");

      double[] ihSums = new double[this.numHidden]; // Scratch
      double[] ihOutputs = new double[this.numHidden];
      double[] hoSums = new double[this.numOutput];

      for (int i = 0; i < xValues.Length; ++i) // xValues to inputs
        this.inputs[i] = xValues[i];

      Console.WriteLine("input-to-hidden weights:");
      FeedForwardProgram.ShowMatrix(this.ihWeights, -1);

      for (int j = 0; j < numHidden; ++j)  // Input-to-hidden weighted sums
        for (int i = 0; i < numInput; ++i)
          ihSums[j] += this.inputs[i] * ihWeights[i][j];

      Console.WriteLine("input-to-hidden sums before adding i-h biases:");
      FeedForwardProgram.ShowVector(ihSums, 2);

      Console.WriteLine("input-to-hidden biases:");
      FeedForwardProgram.ShowVector(this.ihBiases, 2);

      for (int i = 0; i < numHidden; ++i)  // Add biases
        ihSums[i] += ihBiases[i];

      Console.WriteLine("input-to-hidden sums after adding i-h biases:");
      FeedForwardProgram.ShowVector(ihSums, 2);

      for (int i = 0; i < numHidden; ++i)   // Input-to-hidden output
        ihOutputs[i] = LogSigmoid(ihSums[i]);

      Console.WriteLine("input-to-hidden outputs after log-sigmoid activation:");
      FeedForwardProgram.ShowVector(ihOutputs, 2);

      Console.WriteLine("hidden-to-output weights:");
      FeedForwardProgram.ShowMatrix(hoWeights, -1);

      for (int j = 0; j < numOutput; ++j)   // Hidden-to-output weighted sums
        for (int i = 0; i < numHidden; ++i)
          hoSums[j] += ihOutputs[i] * hoWeights[i][j];

      Console.WriteLine("hidden-to-output sums before adding h-o biases:");
      FeedForwardProgram.ShowVector(hoSums, 2);

      Console.WriteLine("hidden-to-output biases:");
      FeedForwardProgram.ShowVector(this.hoBiases, 2);

      for (int i = 0; i < numOutput; ++i)  // Add biases
        hoSums[i] += hoBiases[i];

      Console.WriteLine("hidden-to-output sums after adding h-o biases:");
      FeedForwardProgram.ShowVector(hoSums, 2);

      for (int i = 0; i < numOutput; ++i)   // Hidden-to-output result
        this.outputs[i] = LogSigmoid(hoSums[i]);

      double[] result = new double[numOutput]; // Copy to this.outputs
      this.outputs.CopyTo(result, 0);

      return result;
    }

    private static double LogSigmoid(double z) 
    {
      if (z < -20.0) return 0.0;
      else if (z > 20.0) return 1.0;
      else return 1.0 / (1.0 + Math.Exp(-z));
    }
  } // Class
} // ns
