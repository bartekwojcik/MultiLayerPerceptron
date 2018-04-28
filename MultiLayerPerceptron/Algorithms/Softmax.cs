using System;
using System.Linq;

namespace MultiLayerPerceptron.Algorithms
{
    public class Softmax : IAlgorithm
    {
        private readonly int _numberOfVectors;

        public Softmax(int numberOfVectors)
        {
            _numberOfVectors = numberOfVectors;
        }

        public double[] ProcessDeltaOs(double[] outputs, double[] tagets)
        {
            var deltaOs = new double[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
            {
                var output = outputs[i];
                var target = tagets[i];
                deltaOs[i] = ((output - target) * (output * (-output) + output)) / _numberOfVectors;
            }

            return deltaOs;
        }

        public double[] ProcessOutputs(double[] outputs)
        {
            var exp = outputs.Select(Math.Exp);
            var normaliser = exp.Sum();
            var result = outputs.Select(x => Math.Exp(x) / normaliser);
            return result.ToArray();
        }
    }
}
