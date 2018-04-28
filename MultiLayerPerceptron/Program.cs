using System;
using System.Threading.Tasks;
using MultiLayerPerceptron.Algorithms;

namespace MultiLayerPerceptron
{
    class Program
    {
        static void Main(string[] args)
        {
            int interation = 5000;
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
            
            //=SIN(2*PI()*A10)+COS(4*PI()*A10)*RAND()*0.2
            // where As are numbers from liner space between 0 and 1
            var RandomInputs = new double[,]
            {
                {0},
                {0.038461538},
                {0.076923077},
                {0.115384615},
                {0.153846154},
                {0.192307692},
                {0.230769231},
                {0.269230769},
                {0.307692308},
                {0.346153846},
                {0.384615385},
                {0.423076923},
                {0.461538462},
                {0.5},
                {0.538461538},
                {0.576923077},
                {0.615384615},
                {0.653846154},
                {0.692307692},
                {0.730769231},
                {0.769230769},
                {0.807692308},
                {0.846153846},
                {0.884615385},
                {0.923076923},
                {0.961538462},
                {1}

            };
            var randomTargets = new double[]
            {
                0.006680055,
                0.265486149,
                0.483279114,
                0.668364274,
                0.763504244,
                0.038461538,
                0.798585083,
                0.845324964,
                0.793562162,
                0.759179583,
                0.686347672,
                0.491122895,
                0.243967266,
                0.019711415,
                -0.137190041,
                -0.44675456,
                -0.642939788,
                -0.883257236,
                -0.970691029,
                -1.076569246,
                -1.083657338,
                -0.975380275,
                -0.872895307,
                -0.64233422,
                -0.363677106,
                -0.063115857,
                0.186050921,
            };
            var randomTargets1 = (double[])randomTargets.Clone();

            var softmax = new Softmax(inputs.GetLength(0));
            var logistic = new Logistic(beta);
            var linear = new Linear(randomTargets1.Length);

            var xorPerc = new MultiLayerPerceptron(inputs, (double[])xortargets.Clone(), neurons, beta, momentum, linear);
            var orPerc = new MultiLayerPerceptron(inputs, (double[])ortargets.Clone(), neurons, beta, momentum, linear);
            var andPerc = new MultiLayerPerceptron(inputs, (double[])andtargets.Clone(), neurons, beta, momentum, linear);
            var randomPerc = new MultiLayerPerceptron(RandomInputs, randomTargets, neurons, beta, momentum, new Softmax(randomTargets.GetLength(0)));

            var tasks = new Task[]
            {
                Task.Factory.StartNew(() => xorPerc.Train(interation, eta)),
                Task.Factory.StartNew(() => orPerc.Train(interation, eta)),
                Task.Factory.StartNew(() => andPerc.Train(interation, eta)),
                Task.Factory.StartNew(() => randomPerc.Train(interation, eta))
            };

            Console.Out.WriteLine("Start" + Environment.NewLine);
            Task.WaitAll(tasks);
            Console.Out.WriteLine("Stop" + Environment.NewLine);

            xorPerc.Train(interation, eta);

            Console.WriteLine("XOR");
            xorPerc.ConfusionMatrix(inputs, xortargets);
            Console.Out.WriteLine(Environment.NewLine);

            Console.WriteLine("OR");
            orPerc.ConfusionMatrix(inputs, ortargets);
            Console.Out.WriteLine(Environment.NewLine);

            Console.WriteLine("AND");
            andPerc.ConfusionMatrix(inputs, andtargets);
            Console.Out.WriteLine(Environment.NewLine);

            Console.Out.WriteLine("MY random shit function from excel");
            randomPerc.ConfusionMatrix(RandomInputs, randomTargets1);
            Console.Out.WriteLine(Environment.NewLine);

            Console.ReadKey();
        }
    }
}
