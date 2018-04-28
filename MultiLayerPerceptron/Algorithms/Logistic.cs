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
                //here is something wrong
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
                else if (double.IsInfinity(deltaOs[i]))
                {
                    throw new Exception();
                }
            }

            return deltaOs;
        }

        public double[] ProcessOutputs(double[] outputs)
        {   //here is something wrong
            var result = new double[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
            {
                result[i] = 1 / (1 + Math.Exp(-_beta * outputs[i]));
                var infinity = result[i];
                if (double.IsNegativeInfinity(infinity))
                {
                    result[i] = double.MinValue;
                }
                else if (double.IsPositiveInfinity(infinity))
                {
                    result[i] = double.MaxValue;
                }
            }

            return outputs;
        }

        
    }
}
