using System;

namespace MultiLayerPerceptron.Algorithms
{
    public class Logistic : IAlgorithm
    {
        private readonly double _beta;

        public Logistic(double beta)
        {
            _beta = beta;
        }

        public double[] ProcessDeltaOs(double[] outputs, double[] tagets)
        {
            var deltaOs = new double[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
            {
                deltaOs[i] = _beta * (outputs[i] - tagets[i]) * outputs[i] * (1 - outputs[i]);
                var infinity = deltaOs[i];
                if (double.IsNegativeInfinity(infinity))
                {
                    deltaOs[i] = double.MinValue;
                }
                else if (double.IsPositiveInfinity(infinity))
                {
                    deltaOs[i] = double.MaxValue;
                }
            }

            return deltaOs;
        }

        public double[] ProcessOutputs(double[] outputs)
        {
            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i] = 1 / (1 * Math.Exp(-_beta * outputs[i]));
                var infinity = outputs[i];
                if (double.IsNegativeInfinity(infinity))
                {
                    outputs[i] = double.MinValue;
                }
                else if (double.IsPositiveInfinity(infinity))
                {
                    outputs[i] = double.MaxValue;
                }
            }

            return outputs;
        }

        
    }
}
