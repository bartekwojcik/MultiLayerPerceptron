using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp3
{
    public static class ArrayHelper
    {
        private static Random random;

        static ArrayHelper()
        {
            random = new Random();
        }
        public static double[,] SetRandomWeights(double[,] hiddenWeights,int numberOfNeurons)
        {
            for (int i = 0; i < hiddenWeights.GetLength(0); i++)
            {
                for (int j = 0; j < hiddenWeights.GetLength(1); j++)
                {
                    var randValue = GetRandForWeight(numberOfNeurons);
                    hiddenWeights[i, j] = randValue;
                }
            }

            return hiddenWeights;
        }

        public static double GetRandForWeight(int numberOfNeurons)
        {
            var rand = random.NextDouble();
            rand = rand - 0.5;
            rand = rand * 2;
            rand = rand / Math.Sqrt(numberOfNeurons);
            return rand;
            //var rand = (double)random.Next(10, 100);
            //rand = rand / 1000.0;
            //var randWeight = rand - 0.05;
            //return randWeight;
        }

        public static double[] SetRandomWeights(double[] weights, int numberOfNeurons)
        {
            var random = new Random();
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = GetRandForWeight(numberOfNeurons);
            }

            return weights;
        }

        public static double[,] AddBiasInput(double[,] inputs)
        {
            var localInputs = new double[inputs.GetLength(0), inputs.GetLength(1) + 1];
            for (int i = 0; i < localInputs.GetLength(0); i++)
            {
                for (int j = 0; j < localInputs.GetLength(1); j++)
                {
                    if (j == localInputs.GetLength(1) - 1)
                    {
                        localInputs[i, j] = -1;
                    }
                    else
                    {
                        localInputs[i, j] = inputs[i, j];
                    }
                }
            }
            return localInputs;
        }

        public static double[] AddBiasInput(double[] inputs)
        {
            var localInputs = new double[inputs.Length + 1];
            for (int j = 0; j < localInputs.Length; j++)
            {
                if (j == localInputs.Length - 1)
                {
                    localInputs[j] = -1;
                }
                else
                {
                    localInputs[j] = inputs[j];
                }
            }

            return localInputs;
        }

        public static int RowLength(this double[,] array)
        {
            return array.GetLength(0);
        }

        public static int ColumnLength(this double[,] array)
        {
            return array.GetLength(1);
        }

        public static double[] GetRow(this double[,] trainingInputs, int rowIndex)
        {
            var result = new double[trainingInputs.GetLength(1)];
            for (int i = 0; i < trainingInputs.GetLength(1); i++)
            {
                result[i] = trainingInputs[rowIndex, i];
            }
            return result;
        }

        public static double[] GetColumn(this double[,] array, int columnIndex)
        {
            var result = new double[array.GetLength(0)];
            for (int i = 0; i < array.GetLength(0); i++)
            {
                result[i] = array[i, columnIndex];
            }

            return result;
        }

    }

    public static class MathHelper
    {
        public static double Sigmoid(double value, double beta)
        {
            var result = 1 / (1 + Math.Exp(-beta * value));
            return result;
        }
    }
}
