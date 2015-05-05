using System;
namespace Perceptrons
{
  class PerceptronProgram
  {
    static void Main(string[] args)
    {
      try
      {
        Console.WriteLine("\nBegin Perceptron demo \n");
        int[][] trainingData = new int[5][];
        trainingData[0] = new int[] { 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0 };  // 'A'
        trainingData[1] = new int[] { 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1 };  // 'B'
        trainingData[2] = new int[] { 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0 };  // 'C'
        trainingData[3] = new int[] { 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0 };  // 'D'
        trainingData[4] = new int[] { 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0 };  // 'E'

        Console.WriteLine("Training data input is patterns for characters A-E");
        Console.WriteLine("Goal is to predict patterns that represent 'B'");

        Console.Write("\nTraining input patterns (in row-col");
        Console.WriteLine(" descriptive format):\n");
        ShowData(trainingData[0]);
        ShowData(trainingData[1]);
        ShowData(trainingData[2]);
        ShowData(trainingData[3]);
        ShowData(trainingData[4]);

        Console.Write("\n\nFinding best weights and bias for");
        Console.WriteLine(" a 'B' classifier perceptron");
        int maxEpochs = 1000; //Limit the number of iterations by the main processing loop that is determining the best weights and bias values
        double alpha = 0.075; //learning rate
        double targetError = 0.0; //alternative loop exit criterion

        double bestBias = 0.0; //Stores the Single best bias for the single neuron
        double[] bestWeights = FindBestWeights(trainingData, maxEpochs, alpha, targetError, out bestBias); //Calculates the best set of weights for the neuron. one weight for each input
        Console.WriteLine("\nTraining complete");

        Console.WriteLine("\nBest weights and bias are:\n");
        ShowVector(bestWeights);
        Console.WriteLine(bestBias.ToString("F3"));

        double totalError = TotalError(trainingData, bestWeights, bestBias); //Gets the Error rate from the training data
        Console.Write("\nAfter training total error = ");
        Console.WriteLine(totalError.ToString("F4"));

        int[] unknown = new int[] { 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0 };  // damaged 'B' in 2 positions
        Console.Write("\nPredicting is a 'B' (yes = 1, no = 0)");
        Console.WriteLine(" for the following pattern:\n");
        ShowData(unknown);

        int prediction = Predict(unknown, bestWeights, bestBias); //Predicts what the unknown value is using the best weights and bias
        Console.Write("\nPrediction is " + prediction);
        Console.Write(" which means pattern ");
        string s0 = "is NOT recognized as a 'B'";
        string s1 = "IS recognized as a 'B'";
        if (prediction == 0) Console.WriteLine(s0);
        else Console.WriteLine(s1);

        Console.WriteLine("\nEnd Perceptron demo\n");
        Console.ReadLine();
      }
      catch (Exception ex)
      {
        Console.WriteLine(ex.Message);
        Console.ReadLine();
      }
    } // Main

    public static int StepFunction(double x) 
    {
      if(x>0.5) return 1; //if x is > 0.5 then the neuron fires or is activated
      else return 0; //else it does not fire or activate
    }

    public static int ComputeOutput(int[] trainVector, double[] weights, double bias) 
    {
      Console.WriteLine("Computing Output");
      double dotP = 0.0;
      for(int j = 0; j < trainVector.Length - 1; j++) //trainVector assumes that there is an extra value for the label so it subtracts one from the total so the value is not included in the sum 
        dotP += (trainVector[j] * weights[j]); //sums of all the inputs * the weights of each input
      dotP += bias; //Add the bias to the total sum of the weights
      return StepFunction(dotP); //This is the activation Function
    }

    public static int Predict(int[] dataVector, double[] bestWeights, double bestBias) 
    {
      //This function is almost the same as computeOutput except it does not have and embedded desired value
      double dotP = 0.0;
      for(int j = 0; j < dataVector.Length; j++) 
        dotP += (dataVector[j] * bestWeights[j]); //sums of all the inputs * the weights of each input
      dotP += bestBias; //Add the bias to the total sum of the weights
      return StepFunction(dotP); //This is the activation Function
    }

    public static double TotalError(int[][] trainingData, double[] weights, double bias) 
    { 
      double sum = 0.0; //sum of the error
      for (int i = 0; i < trainingData.Length; ++i) //looping though the training set
      {
        int desired = trainingData[i][trainingData[i].Length - 1]; //gets the desired output
        int output = ComputeOutput(trainingData[i], weights, bias);//computes the output with the wights and bias
        sum += (desired - output) * (desired - output); //sum of the squared differences between the ouput and the computed output
      }
      return 0.5 * sum; //one half of the sum
    }

    public static double[] FindBestWeights(int[][] trainingData, int maxEpochs, double alpha, double targetError, out double bestBias) 
    {
      int dim = trainingData[0].Length - 1; //size of the data array without the label
      double[] weights = new double[dim]; //Creates an array for the weights of size dim for each input initalized to 0.0
      double bias = 0.05; //arbitrary value to initalize the bias value
      double totalError = double.MaxValue; //setting totalError to max value a double can be
      int epoch = 0; //Counter for the loop epoch

      while(epoch < maxEpochs && totalError > targetError) //Loop exits either if epoch is < maxEpochs or total error is > targetError
      {
        for (int i = 0; i < trainingData.Length; i++) //Looping though all the training sets
        {
          int desired = trainingData[i][trainingData[i].Length - 1]; //gets the expected label of the training set
          int output = ComputeOutput(trainingData[i], weights, bias); //computes the output of the training set
          int delta = desired - output; //gets the delta value either 1 or -1 if the prediction was wrong and 0 if it was right

          for (int j = 0; j < weights.Length; j++) //Looping though all the weights
            weights[j] = weights[j]+(alpha * delta * trainingData[i][j]); //adjusting the weights for each input

          bias = bias + (alpha * delta); //adjusting the bias for the neuron
        }

        totalError = TotalError(trainingData, weights, bias);//calculating the total Error with the current Weights and Bias
        epoch++; //Increasing count
      }
      bestBias = bias; //Setting best Bias
      return weights;
    }

    public static void ShowVector(double[] vector) 
    { 
      for (int i = 0; i < vector.Length; i++)
      {
        if (i > 0 && i % 4 == 0) Console.WriteLine("");
        Console.Write(vector[i].ToString("+0.000;-0.000") + " ");
          }
        Console.WriteLine("");
    }

    public static void ShowData(int[] data) 
    {
      for (int i = 0; i < 20; ++i)  // Hardcoded to indicate custom
      {
        if (i % 4 == 0)
        {
          Console.WriteLine("");
          Console.Write(" ");
        }
        if (data[i] == 0) Console.Write(" ");
        else Console.Write("1");
      }
      Console.WriteLine("");
    }

  } // class
} // ns
