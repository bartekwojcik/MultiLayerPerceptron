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
            int interation = 50000;
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

            var perceptron = new MultiLayerPerceptron(inputs, xortargets, neurons, beta: 1, momentum: 0.9, ouTtype: FunType.Logistic);
            var perceptron1 = new MultiLayerPerceptron(inputs, ortargets, neurons, beta: 1, momentum: 0.9, ouTtype: FunType.Logistic);
            var perceptron2 = new MultiLayerPerceptron(inputs, andtargets, neurons, beta: 1, momentum: 0.9, ouTtype: FunType.Logistic);

            var tasks = new Task[]
            {
                Task.Factory.StartNew(() => perceptron.Train(interation, eta)),
                Task.Factory.StartNew(() => perceptron1.Train(interation, eta)),
                Task.Factory.StartNew(() => perceptron2.Train(interation, eta))
            };

            Console.Out.WriteLine("Start" + Environment.NewLine);
            Task.WaitAll(tasks);
            Console.Out.WriteLine("Stop" + Environment.NewLine);
            Console.WriteLine("XOR");
            perceptron.ConfusionMatrix(inputs, xortargets);
            Console.Out.WriteLine(Environment.NewLine);
            Console.WriteLine("OR");
            perceptron1.ConfusionMatrix(inputs, ortargets);
            Console.Out.WriteLine(Environment.NewLine);
            Console.WriteLine("AND");
            perceptron2.ConfusionMatrix(inputs, andtargets);

            Console.ReadKey();
        }
    }
}
