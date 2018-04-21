using System;
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
            int interation = 10000;
            double eta = 0.25;
            int neurons = 5;
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

            var xorPerc = new MultiLayerPerceptron(inputs, xortargets, neurons, beta: 1, momentum: 0.9, ouTtype: FunType.Logistic);
            var orPerc = new MultiLayerPerceptron(inputs, ortargets, neurons, beta: 1, momentum: 0.9, ouTtype: FunType.Logistic);
            var andPerc = new MultiLayerPerceptron(inputs, andtargets, neurons, beta: 1, momentum: 0.9, ouTtype: FunType.Logistic);

            var tasks = new Task[]
            {
                Task.Factory.StartNew(() => xorPerc.Train(interation, eta)),
                Task.Factory.StartNew(() => orPerc.Train(interation, eta)),
                Task.Factory.StartNew(() => andPerc.Train(interation, eta))
            };

            Console.Out.WriteLine("Start" + Environment.NewLine);
            Task.WaitAll(tasks);
            Console.Out.WriteLine("Stop" + Environment.NewLine);

            Console.WriteLine("XOR");
            xorPerc.ConfusionMatrix(inputs, xortargets);
            Console.Out.WriteLine(Environment.NewLine);

            Console.WriteLine("OR");
            orPerc.ConfusionMatrix(inputs, ortargets);
            Console.Out.WriteLine(Environment.NewLine);

            Console.WriteLine("AND");
            andPerc.ConfusionMatrix(inputs, andtargets);

            Console.ReadKey();
        }
    }
}
