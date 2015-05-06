using System;
namespace PerceptronClassification
{
  class PerceptronClassificationProgram
  {
    static void Main(string[] args)
    {
      try
      {
        Console.WriteLine("\nBegin perceptron classification demo");
        double[][] trainData = new double[16][];
        trainData[0] = new double[] { -5.0, -5.5 };
        trainData[1] = new double[] { -3.5, -6.0 };
        trainData[2] = new double[] { -2.0, -4.5 };
        trainData[3] = new double[] { -2.0, -1.5 };
        trainData[4] = new double[] { -3.5, 2.0 };
        trainData[5] = new double[] { -2.5, 4.0 };
        trainData[6] = new double[] { -1.5, 3.0 };
        trainData[7] = new double[] { 2.0, 3.5 };
        trainData[8] = new double[] { 4.5, 5.0 };
        trainData[9] = new double[] { 6.0, 2.5 };
        trainData[10] = new double[] { 3.0, 1.5 };
        trainData[11] = new double[] { 1.5, -5.0 };
        trainData[12] = new double[] { 2.0, 2.0 };
        trainData[13] = new double[] { 3.5, -4.0 };
        trainData[14] = new double[] { 4.0, -5.5 };
        trainData[15] = new double[] { 4.5, -2.0 };

        int[] Y = new int[16] { -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1 };

        Console.WriteLine("\nTraining data: \n");
        ShowTrainData(trainData, Y);

        double[] weights = null;
        double bias = 0.0;
        double alpha = 0.001;
        int maxEpochs = 500;

        Console.Write("\nSetting learning rate to " + alpha.ToString("F3"));
        Console.WriteLine(" and maxEpochs to " + maxEpochs);

        Console.WriteLine("\nBeginning training the perceptron");
        Train(trainData, alpha, maxEpochs, Y, out weights, out bias);
        Console.WriteLine("Training complete");

        Console.WriteLine("\nBest percetron weights found: ");
        ShowVector(weights, 4);
        Console.Write("\nBest perceptron bias found = ");
        Console.WriteLine(bias.ToString("F4"));

        double acc = Accuracy(trainData, weights, bias, Y);
        Console.Write("\nAccuracy of perceptron on training data = ");
        Console.WriteLine(acc.ToString("F2"));

        double[] unknown = new double[] { 1.0, 4.5 };
        Console.WriteLine("\nNew data with unknown class = ");
        ShowVector(unknown, 1);
        Console.WriteLine("\nUsing best weights and bias to classify data");
        int c = ComputeOutput(unknown, weights, bias);
        Console.Write("\nPredicted class of new data = ");
        Console.WriteLine(c.ToString("+0;-0"));

        Console.WriteLine("\nEnd perceptron demo\n");
        Console.ReadLine();
      }
      catch (Exception ex)
      {
        Console.WriteLine(ex.Message);
        Console.ReadLine();
      }
    } // Main

    static int ComputeOutput(double[] data, double[] weights, double bias) 
    {
      double result = 0.0;
      for(int j = 0; j < data.Length; j++)
      {
        result += data[j] * weights[j];
      }
      result += bias;
      return Activation(result);
    }

    static int Activation(double x) 
    {
      if (x >= 0.0) 
      {
        return +1;
      }
      else
      {
        return -1;
      }
    }

    static double Accuracy(double[][] trainData, double[] weights, double bias, int[] Y) 
    {
      int numCorrect = 0;
      int numWrong = 0;
      for (int i = 0; i < trainData.Length; ++i)
      {
        int output = ComputeOutput(trainData[i], weights, bias);
        if (output == Y[i]) ++numCorrect;
        else ++numWrong;
      }
      return (numCorrect * 1.0) / (numCorrect + numWrong);
    }

    static double TotalError(double[][] trainData, double[] weights, double bias, int[] Y) 
    { 
      //Sum of all the Errors
      double totErr = 0.0;
      for (int i = 0; i < trainData.Length; i++)
      {
        totErr += Error(trainData[i], weights, bias, Y[i]);
      }
      return totErr;
    }

    static double Error(double[] data, double[] weights, double bias, int Y) 
    {
      //Calculates the mesures how far away a perceotrin's pre-activation output for a single training data item is from the aactual class value
      //Error = one-half of the sum of the squared deviation
      double sum = 0.0;
      for (int j = 0; j < data.Length; j++)
      {
        sum += data[j] * weights[j];
      }
      sum += bias;
      return 0.5 * (sum-Y)*(sum-Y);
    }

    static void Train(double[][] trainData, double alpha, int maxEpochs, int[] Y, out double[] weights, out double bias) 
    {
      /*
       *loop until done
       *  foreach training data item
       *    compute output using weights and bias
       *    if the output is incorrect then
       *      adjust weights and bias
       *      compute error
       *      if error < smallest error so far
       *        smallest error so far = error
       *        save new weights and bias
       *      end if
       *    end if
       *    increment loop counter
       *  end foreach
       *end loop
       *return best weights and bias values found
       */
      int numWeights = trainData[0].Length;

      double[] bestWeights = new double[numWeights];
      weights = new double[numWeights];
      double bestBias = 0.0;
      bias = 0.01;
      double bestError = double.MaxValue;
      int epoch = 0;

      while(epoch < maxEpochs)
      {
        for (int i = 0; i < trainData.Length; i++)
        {
          int output = ComputeOutput(trainData[i], weights, bias);
          int desired = Y[i];

          if(output != desired)
          {
            double delta = desired - output;
            for(int j = 0; j < numWeights; j++)
            {
              weights[j] = weights[j] + (alpha * delta * trainData[i][j]);
            }

            bias = bias + (alpha * delta);

            double totalError = TotalError(trainData, weights, bias, Y);
            if(totalError < bestError)
            {
              bestError = totalError;
              Array.Copy(weights, bestWeights, weights.Length);
              bestBias = bias;
            }
          }
        }
        epoch++;
      }
      Array.Copy(bestWeights, weights, bestWeights.Length);
      bias = bestBias;
      return;
    }

    static void ShowVector(double[] vector, int decimals) 
    {
      for (int i = 0; i < vector.Length; ++i)
        Console.Write(vector[i].ToString("F" + decimals) + " ");
      Console.WriteLine("");
    }

    static void ShowTrainData(double[][] trainData, int[] Y) 
    {
      for (int i = 0; i < trainData.Length; ++i)
      {
        Console.Write("[" + i.ToString().PadLeft(2, ' ') + "]  ");
        for (int j = 0; j < trainData[i].Length; ++j)
        {
          Console.Write(trainData[i][j].ToString("F1").PadLeft(6, ' '));
        }
        Console.WriteLine("  ->  " + Y[i].ToString("+0;-0"));
      }
    }

  } // class
} // ns
