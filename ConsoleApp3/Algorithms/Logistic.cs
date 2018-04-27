using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp3.Algorithms
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
            }

            return deltaOs;
        }

        public double[] ProcessOutputs(double[] outputs)
        {
            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i] = 1 / (1 * Math.Exp(-_beta * outputs[i]));
            }

            return outputs;
        }

        
    }

    public interface IAlgorithm
    {
        double[] ProcessDeltaOs(double[] outputs, double[] tagets);
        double[] ProcessOutputs(double[] outputs);
    }
}
