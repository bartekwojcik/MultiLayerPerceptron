﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp3
{
    class Program
    {
        static void Main(string[] args)
        {
            var xor = new double[,]
            {
                {0, 0},
                {1, 0},
                {0, 1},
                {1, 1},
            };
            var targets = new double[]
            {
                1, 0, 0, 1
            };

            var perceptron = new MultiLayerPerceptron(xor, targets, beta: 1, momentum: 0.9, ouTtype:FunType.Logistic);
            perceptron.Train(100, 0.25);
            perceptron.ConfusionMatrix(xor, targets);

            Console.ReadKey();
        }
    }
}
