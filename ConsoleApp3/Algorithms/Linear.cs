using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp3.Algorithms
{
    public class Linear : IAlgorithm
    {
        private readonly int _numberOrVectors;

        public Linear(int numberOrVectors)
        {
            _numberOrVectors = numberOrVectors;
        }
        public double[] ProcessDeltaOs(double[] outputs, double[] tagets)
        {
            var deltaOs = new double[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
            {
                deltaOs[i] = (outputs[i] - tagets[i]) / _numberOrVectors;
            }

            return deltaOs;
        }

        public double[] ProcessOutputs(double[] outputs)
        {
            return outputs;
        }
    }
}
