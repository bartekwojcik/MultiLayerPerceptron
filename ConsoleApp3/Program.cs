using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using ConsoleApp3.Algorithms;

namespace ConsoleApp3
{
    class Program
    {
        static void Main(string[] args)
        {
            int interation = 500;
            double eta = 0.25;
            int neurons = 5;
            var beta = 1;
            var momentum = 0.9;
            var inputs = new double[,]
            {
                {0, 0},
                {1, 0},
                {0, 1},
                {1, 1},
            };
            var xortargets = new double[]
            {
                0, 1, 1, 0
            };
            var ortargets = new double[]
            {
                0, 1, 1, 1
            };
            var andtargets = new double[]
            {
                0, 0, 0, 1
            };

            var ortatgets1 = new double[ortargets.Length];
            ortargets.CopyTo(ortatgets1,0);

            var andtargets1 = new double[andtargets.Length];
            andtargets.CopyTo(andtargets1, 0);

            var xortargets1 = new double[xortargets.Length];
            xortargets.CopyTo(xortargets1, 0);

            var algorithm = new Logistic(beta);

            var xorPerc = new MultiLayerPerceptron(inputs, xortargets, neurons, beta, momentum, algorithm);
            var orPerc = new MultiLayerPerceptron(inputs, ortargets, neurons, beta, momentum, algorithm);
            var andPerc = new MultiLayerPerceptron(inputs, andtargets, neurons, beta, momentum, algorithm);

            var tasks = new Task[]
            {
                Task.Factory.StartNew(() => xorPerc.Train(interation, eta)),
                Task.Factory.StartNew(() => orPerc.Train(interation, eta)),
                Task.Factory.StartNew(() => andPerc.Train(interation, eta))
            };

            Console.Out.WriteLine("Start" + Environment.NewLine);
            Task.WaitAll(tasks);
            Console.Out.WriteLine("Stop" + Environment.NewLine);

            //xorPerc.Train(interation,eta);

            Console.WriteLine("XOR");
            xorPerc.ConfusionMatrix(inputs, xortargets1);
            Console.Out.WriteLine(Environment.NewLine);

            Console.WriteLine("OR");
            orPerc.ConfusionMatrix(inputs, ortatgets1);
            Console.Out.WriteLine(Environment.NewLine);

            Console.WriteLine("AND");
            andPerc.ConfusionMatrix(inputs, andtargets1);

            Console.ReadKey();
        }
    }
}
