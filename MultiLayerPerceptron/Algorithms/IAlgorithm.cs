﻿namespace MultiLayerPerceptron.Algorithms
{
    public interface IAlgorithm
    {
        double[] ProcessDeltaOs(double[] outputs, double[] tagets);
        double[] ProcessOutputs(double[] outputs);
    }
}