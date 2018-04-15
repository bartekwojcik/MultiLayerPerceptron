using System;
using System.Collections.Generic;
using System.Linq;
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
        private double[,] _trainingInput;
        private double[] _trainingTargets;
        private int _beta;
        private double _momentum;
        private FunType _ouTtype;
        private int _numberOfHiddenNeurons;
        private double[,] _hiddenWeights;
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

            this._hiddenWeights = new double[numberOfHiddenNeurons, _trainingInput.GetLength(1)];
            this._outputWeights = new double[numberOfHiddenNeurons + 1];

            _hiddenWeights = ArrayHelper.SetRandomWeights(_hiddenWeights);
            _outputWeights = ArrayHelper.SetRandomWeights(_outputWeights);

            this._lastOutputUpdates = new double[_outputWeights.Length];
            this._lastHiddenUpdates = new double[numberOfHiddenNeurons, _trainingInput.GetLength(1)];
        }


        public void Train(int iterations, double eta)
        {
            for (int i = 0; i < iterations; i++)
            {
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
                var output = outputs[i];
                var target = _trainingTargets[i];
                var deltaO = _beta * (output - target) * output * (1 - output);
                deltasOs[i] = deltaO;
                for (int j = 0; j < _outputWeights.Length; j++)
                {
                    var neuron = hiddenResults[i, j];
                    var update = eta * deltaO * neuron;
                    var currentValue = _outputWeights[j];
                    var wholeUpdate = update + _momentum * _lastOutputUpdates[j];
                    _outputWeights[j] = currentValue - wholeUpdate;
                    _lastOutputUpdates[j] = update;
                }
            }

            //here is the highest chance of mistake
            var test = new string[_numberOfHiddenNeurons, _hiddenWeights.GetLength(1)];
            var deltasHs = new Double[_numberOfHiddenNeurons, _hiddenWeights.GetLength(1)];
            for (int i = 0; i < _hiddenWeights.GetLength(0); i++)
            {
                for (int j = 0; j < _hiddenWeights.GetLength(1); j++)
                {
                    var sum = 0.0;
                    var neuron = hiddenResults[i, j];

                    var deltaO = deltasOs[j];

                    for (int k = 0; k < _trainingInput.GetLength(1); k++)
                    {
                        var weight = _hiddenWeights[k, j];
                        sum += deltaO * weight;

                    }


                    var deltaH = neuron * (1 - neuron) * sum;
                    deltasHs[i, j] = deltaH;

                    var update = eta * deltaH * _trainingInput[i, j];
                    var currentValue = _hiddenWeights[i, j];
                    var wholeUpdate = update + _momentum * _lastHiddenUpdates[i, j];
                    _hiddenWeights[i, j] = currentValue - wholeUpdate;
                    _lastHiddenUpdates[i, j] = update;
                }
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
            var test = new string[inputs.GetLength(0)];
            for (int i = 0; i < outputs.Length; i++)
            {
                for (int k = 0; k < _outputWeights.Length; k++)
                {
                    var neuron = hiddenNeuronsValues[i, k];
                    var weight = _outputWeights[k];
                    outputs[i] += neuron * weight;
                    test[i] += $"N{i + 1}{k + 1}*w{k + 1} +";
                }


                var sigmoidValue = MathHelper.Sigmoid(outputs[i], _beta);
                outputs[i] = sigmoidValue;
            }
            return outputs;
        }

        private double[,] CalculateHiddenLayer(double[,] inputs)
        {
            var hiddenLayerNauronValues = new double[inputs.GetLength(0), _numberOfHiddenNeurons];
            var test = new string[inputs.GetLength(0), _numberOfHiddenNeurons];

            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                var row = inputs.GetRow(i);
                for (int j = 0; j < _numberOfHiddenNeurons; j++)
                {

                    for (int k = 0; k < row.Length; k++)
                    {
                        for (int l = 0; l < _hiddenWeights.GetLength(0); l++)
                        {
                            var input = row[k];
                            var weight = _hiddenWeights[j, k];
                            var delta = input * weight;
                            if (l == j)
                            {
                                test[i, j] += $"x{k + 1}*w{j + 1}{k + 1} +";
                                hiddenLayerNauronValues[i, j] += delta;
                            }
                        }
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

                line += $"| target: {targets[i]}, output: {outputs[i]}";
                Console.Out.WriteLine(line);
            }

            Console.ReadKey();
        }

        public double EarlyStopping()
        {
            throw new NotImplementedException();
        }

        //run forward
        // run backward
    }
}
