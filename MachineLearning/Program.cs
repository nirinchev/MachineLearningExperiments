
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace MachineLearning
{
	public class Program
	{
		private const int Features = 4;
		private const double Alpha = 0.1;
		private const double Lambda = 100;
		private const int TestSize = 100;

		private static readonly Random rand = new Random();
		private static readonly int[] TrainingDataSizes =
		{
			10,
			100,
			1000,
			5000
		};

		static void Main(string[] args)
		{
			var input = ReadInput();

			var sw = new Stopwatch();
			foreach (var dataSize in TrainingDataSizes)
			{
				sw.Start();
				var featureData = input.Item1.SubMatrix(0, dataSize, 0, input.Item1.ColumnCount);
				var results = input.Item2.SubVector(0, dataSize);

				var theta = CalculateTheta(featureData, results);

				var test = 0.0;
				for (var i = 0; i < TestSize; i++)
				{
					var index = rand.Next(TestSize, input.Item1.RowCount);
					var testData = input.Item1.Row(index);
					test += Math.Pow(Hypothesis(theta, testData) - input.Item2[index], 2);
				}

				sw.Stop();

				Console.WriteLine($"Size: {dataSize}, PredictionCost: {Math.Sqrt(test / TestSize)}, Elapsed: {sw.ElapsedMilliseconds / 1000.0} s");
				sw.Reset();
			}

			Console.Read();
		}

		private static Vector<double> CalculateTheta(Matrix<double> featureData, Vector<double> results)
		{
			var theta = Vector<double>.Build.Dense(featureData.ColumnCount, 0);
			var cost = Cost(theta, featureData, results);
			double costDifference;
			do
			{
				theta = UpdateTheta(theta, featureData, results);
				var newCost = Cost(theta, featureData, results);
				costDifference = cost - newCost;
				cost = newCost;
			}
			while (costDifference > 0.001);
			return theta;
		}

		private static Tuple<Matrix<double>, Vector<double>> ReadInput()
		{
			var file = File.ReadAllLines("Data.csv").Skip(1)
													.Select(s => s.Split(',')
																  .Select(double.Parse)
																  .ToArray())
													.ToArray();
			var featureMatrix = Matrix<double>.Build.Dense(file.Length, Features + 1);
			var outputVector = Vector<double>.Build.Dense(file.Length);
			for (var i = 0; i < file.Length; i++)
			{
				var data = file[i];
				var features = new[] { 1.0 }.Concat(data.Take(Features)).ToArray();

				featureMatrix.SetRow(i, features);
				outputVector[i] = data[Features];
			}

			foreach (var column in featureMatrix.EnumerateColumnsIndexed())
			{
				if (column.Item1 == 0)
				{
					continue;
				}
				var min = column.Item2.Minimum();
				var max = column.Item2.Maximum();
				var mean = column.Item2.Average();
				for (var i = 0; i < column.Item2.Count; i++)
				{
					featureMatrix[i, column.Item1] = (featureMatrix[i, column.Item1] - mean) / (max - min);
				}
			}

			var featureCount = featureMatrix.ColumnCount;
			var secondOrderMatrix = Matrix<double>.Build.Dense(featureMatrix.RowCount, 2 * featureCount);
			secondOrderMatrix.SetSubMatrix(0, 0, featureMatrix);
			for (var i = 0; i < featureMatrix.RowCount; i++)
			{
				for (var j = 0; j < featureCount; j++)
				{
					secondOrderMatrix[i, featureCount + j] = Math.Pow(featureMatrix[i, j], 2);
				}
			}

			return Tuple.Create(secondOrderMatrix, outputVector);
		}

		private static double Hypothesis(Vector<double> theta, Vector<double> features) => theta.DotProduct(features);

		private static Vector<double> UpdateTheta(Vector<double> theta, Matrix<double> trainingData, Vector<double> results)
		{
			var result = DenseVector.Create(theta.Count, 0);
			for (var j = 0; j < theta.Count; j++)
			{
				var lambdaTerm = j == 0 ? 0 : Alpha * Lambda / trainingData.RowCount;
				result[j] = theta[j] * (1 - lambdaTerm) - (Alpha / trainingData.RowCount) * trainingData.EnumerateRowsIndexed().Sum(r => (Hypothesis(theta, r.Item2) - results[r.Item1]) * r.Item2[j]);
			}
			return result;
		}

		private static double Cost(Vector<double> theta, Matrix<double> trainingData, Vector<double> results)
		{
			return (1 / (2.0 * trainingData.RowCount)) * (trainingData.EnumerateRowsIndexed().Sum(r => Math.Pow(Hypothesis(theta, r.Item2) - results[r.Item1], 2)) + Lambda * theta.Skip(1).Sum(t => Math.Pow(t, 2)));
		}
	}
}
