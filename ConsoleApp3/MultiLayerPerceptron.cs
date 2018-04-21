using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp3
{
    public enum FunType
    {
        Logistic
    }

    public class MultiLayerPerceptron
    {
        private readonly double[,] _trainingInput;
        private readonly double[] _trainingTargets;
        private int _beta;
        private readonly double _momentum;
        private FunType _ouTtype;
        private readonly int _numberOfHiddenNeurons;
        private readonly double[,] _hiddenWeights;
        private readonly double[] _outputWeights;
        private int _currentIteration;
        private double[] _lastOutputUpdates;
        private double[,] _lastHiddenUpdates;

        public MultiLayerPerceptron(double[,] trainingInput, double[] trainingTargets, int numberOfHiddenNeurons, int beta, double momentum, FunType ouTtype)
        {
            this._beta = beta;
            this._momentum = momentum;
            this._ouTtype = ouTtype;
            this._numberOfHiddenNeurons = numberOfHiddenNeurons;

            this._trainingInput = ArrayHelper.AddBiasInput(trainingInput); // + bias
            this._trainingTargets = trainingTargets;

            this._hiddenWeights = new double[_trainingInput.GetLength(1), numberOfHiddenNeurons];
            this._outputWeights = new double[numberOfHiddenNeurons + 1];

            _hiddenWeights = ArrayHelper.SetRandomWeights(_hiddenWeights, numberOfHiddenNeurons);
            _outputWeights = ArrayHelper.SetRandomWeights(_outputWeights, numberOfHiddenNeurons);


        }


        public void Train(int iterations, double eta)
        {
            _lastHiddenUpdates = new double[_hiddenWeights.RowLength(), _hiddenWeights.ColumnLength() + 1];
            _lastOutputUpdates = new double[_outputWeights.Length];
            for (int i = 0; i < iterations; i++)
            {
                ShuffleRows();
                this._currentIteration = i;
                var result = ForwardPhase(_trainingInput);
                BackwardsPhase(result.OutputResult, result.HiddenValues, eta);
            }
        }

        private void BackwardsPhase(double[] outputs, double[,] hiddenResults, double eta)
        {
            var deltasOs = new double[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
            {
                deltasOs[i] = (outputs[i] - _trainingTargets[i]) * outputs[i] * (1 - outputs[i]);
            }

            var deltasHs = new Double[hiddenResults.RowLength(), hiddenResults.ColumnLength()];
            for (int i = 0; i < deltasHs.RowLength(); i++)
            {
                for (int j = 0; j < deltasHs.ColumnLength(); j++)
                {
                    deltasHs[i, j] = deltasHs[i, j] + hiddenResults[i, j] * deltasOs[i] * _outputWeights[j];
                }
            }

            for (int i = 0; i < _hiddenWeights.RowLength(); i++)
            {
                for (int j = 0; j < _hiddenWeights.ColumnLength(); j++)
                {

                    double fullUpdate = 0;
                    for (int k = 0; k < _trainingInput.RowLength(); k++)
                    {
                        var update = eta * _trainingInput[k, i] * deltasHs[k, j];
                        fullUpdate += update;
                    }
                    var momentum = _momentum * _lastHiddenUpdates[i, j];
                    fullUpdate += momentum;
                    _lastHiddenUpdates[i, j] = fullUpdate;
                    _hiddenWeights[i, j] = _hiddenWeights[i, j] - fullUpdate;
                }
            }

            for (int i = 0; i < _outputWeights.Length; i++)
            {
                double fullUpdate = 0;
                for (int k = 0; k < hiddenResults.RowLength(); k++)
                {
                    var update = eta * hiddenResults[k, i] * deltasOs[k];
                    fullUpdate += update;
                }
                var momentum = _momentum * _lastOutputUpdates[i];
                fullUpdate += momentum;
                _outputWeights[i] = _outputWeights[i] - fullUpdate;
                _lastOutputUpdates[i] = fullUpdate;
            }

        }

        private ForwardPhaseResult ForwardPhase(double[,] inputs)
        {
            var hiddenNeuronsValues = CalculateHiddenLayer(inputs);
            hiddenNeuronsValues = ArrayHelper.AddBiasInput(hiddenNeuronsValues);
            var outputs = CalculateOutputLayerValue(hiddenNeuronsValues, inputs);

            return new ForwardPhaseResult()
            {
                HiddenValues = hiddenNeuronsValues,
                OutputResult = outputs
            };
        }

        private double[] CalculateOutputLayerValue(double[,] hiddenNeuronsValues, double[,] inputs)
        {
            var outputs = new double[inputs.GetLength(0)];

            for (int i = 0; i < hiddenNeuronsValues.RowLength(); i++)
            {
                for (int k = 0; k < _outputWeights.Length; k++)
                {
                    var value = hiddenNeuronsValues[i, k];
                    var weight = _outputWeights[k];
                    var mult = value * weight;
                    outputs[i] = outputs[i] + mult;
                }
                var sigmoidValue = MathHelper.Sigmoid(outputs[i], _beta);
                outputs[i] = sigmoidValue;
            }
            return outputs;
        }

        private double[,] CalculateHiddenLayer(double[,] inputs)
        {
            var hiddenLayerNauronValues = new double[inputs.GetLength(0), _numberOfHiddenNeurons];

            for (int i = 0; i < inputs.RowLength(); i++)
            {
                for (int j = 0; j < _numberOfHiddenNeurons; j++)
                {
                    for (int k = 0; k < inputs.ColumnLength(); k++)
                    {
                        var input = inputs[i, k];
                        var weight = _hiddenWeights[k, j];
                        var mult = input * weight;
                        hiddenLayerNauronValues[i, j] = hiddenLayerNauronValues[i, j] + mult;
                    }
                    var sigmoidValue = MathHelper.Sigmoid(hiddenLayerNauronValues[i, j], _beta);
                    hiddenLayerNauronValues[i, j] = sigmoidValue;
                }
            }

            return hiddenLayerNauronValues;
        }

        public void ConfusionMatrix(double[,] inputs, double[] targets)
        {
            //for now i am just simple priting output and targets
            var biasedInputs = ArrayHelper.AddBiasInput(inputs);
            var forwardPhaseResult = ForwardPhase(biasedInputs);
            var outputs = forwardPhaseResult.OutputResult;
            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                var line = "inputs: ";
                for (int j = 0; j < inputs.GetLength(1); j++)
                {
                    line += $"{inputs[i, j]}, ";
                }

                line += $"| target: {targets[i]}, output: {Math.Round(outputs[i], 4)}";
                Console.Out.WriteLine(line);
            }

        }

        public double EarlyStopping()
        {
            throw new NotImplementedException();
        }



        private void ShuffleRows()
        {
            var dim1 = _trainingInput.GetLength(0);
            var dim2 = _trainingInput.GetLength(1);
            var all = new double[dim1, dim2 + 1];
            var list = new List<double[]>();
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2 + 1; k++)
                {
                    if (k < dim2)
                    {
                        all[j, k] = _trainingInput[j, k];
                    }
                    else
                    {
                        all[j, k] = _trainingTargets[j];
                    }
                }

                var thisRow = all.GetRow(j);
                list.Add(thisRow);
            }

            var shuffled = list.OrderBy(a => Guid.NewGuid()).ToList();

            var newAll = new double[dim1, dim2 + 1];
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2 + 1; k++)
                {
                    newAll[j, k] = shuffled[j][k];
                }
            }

            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2 + 1; k++)
                {
                    if (k < dim2)
                    {
                        _trainingInput[j, k] = newAll[j, k];
                    }
                    else
                    {
                        _trainingTargets[j] = newAll[j, k];
                    }
                }
            }

        }

        //run forward
        // run backward
    }
}
