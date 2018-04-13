using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp3
{
    public enum FunType //todo zmien nazwe
    {
        Logistic
    }

    public class MultiLayerPerceptron
    {
        private double[,] _trainingInput;
        private double[] _trainingTargets;
        private int _beta;
        private double _momentum;
        private FunType _ouTtype;
        private int _numberOfHiddenNeurons;
        private double[] _hiddenWeights;
        private readonly double[] _outputWeights;

        public MultiLayerPerceptron(double[,] trainingInput, double[] trainingTargets, int numberOfHiddenNeurons, int beta, double momentum, FunType ouTtype)
        {

            this._beta = beta;
            this._momentum = momentum;
            this._ouTtype = ouTtype;
            this._numberOfHiddenNeurons = numberOfHiddenNeurons;

            this._trainingInput = AddBiasInput(_trainingInput); // + bias
            this._trainingTargets = trainingTargets;

            this._hiddenWeights = new double[_trainingInput.GetLength(1)];
            this._outputWeights = new double[numberOfHiddenNeurons + 1];

            SetRandomWeights(_hiddenWeights);
            SetRandomWeights(_outputWeights);

        }

        public double[] SetRandomWeights(double[] weights)
        {
            var random = new Random();
            for (int i = 0; i < weights.Length; i++)
            {
                var rand = (double)random.Next(0, 100);
                rand = rand / 1000.0;
                var randWeight = rand - 0.05;
                var rounded = Math.Round(randWeight, 2);
                weights[i] = rounded;
            }

            return weights;
        }

        public void Train(int i, double eta)
        {
            var output = Forward(_trainingInput);
        }

        private double[] Forward(double[,] trainingInputs)
        {
            var hiddenNeuronsValues = CalculateHiddenLayer(trainingInputs);
            
        }

        private double CalculateOutputLayerValue(double[] hiddenNauronsValues)
        {

        }

        private double[] CalculateHiddenLayer(double[,] inputs)
        {
            var hiddenLayerNauronValues = new double[_numberOfHiddenNeurons];
            for (int i = 0; i < _numberOfHiddenNeurons; i++)
            {
                var neuronValue = 0.0;
                for (int j = 0; j < inputs.GetLength(0); j++)
                {
                    var row = GetRow(inputs, j); //{0,1,-1}
                    for (int k = 0; k < row.Length; k++)
                    {
                        var input = row[k]; //1
                        for (int l = 0; l < _hiddenWeights.Length; l++)
                        {
                            var weight = _hiddenWeights[l];
                            var weightInput = weight * input;
                            neuronValue += weightInput;
                        }
                    }
                }
                var sigmoidResult = Sigmoid(neuronValue);
                hiddenLayerNauronValues[i] = sigmoidResult;
            }
            return hiddenLayerNauronValues;
        }

        private double Sigmoid(double value)
        {
            var result = 1 / (1 + Math.Exp(-this._beta * value));
            return result;
        }

        public void ConfusionMatrix(double[,] doubles, double[] targets1)
        {
            throw new NotImplementedException();
        }

        public double EarlyStopping()
        {
            throw new NotImplementedException();
        }

        private static double[,] AddBiasInput(double[,] inputs)
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

        private double[] GetRow(double[,] trainingInputs, int rowIndex)
        {
            var result = new double[trainingInputs.GetLength(1)];
            for (int i = 0; i < trainingInputs.GetLength(1); i++)
            {
                result[i] = trainingInputs[rowIndex, i];
            }
            return result;
        }

        private double[] GetColumn(double[,] array, int columnIndex)
        {
            var result = new double[array.GetLength(0)];
            for (int i = 0; i < array.GetLength(0); i++)
            {
                result[i] = array[i, columnIndex];
            }

            return result;
        }


        //run forward
        // run backward
    }
}
