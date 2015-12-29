using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace ML.Classification
{
	public class Program
	{
		private const double Alpha = 0.1;

		private static readonly Random rand = new Random();
		private static readonly int[] TrainingDataSizes =
		{
			10,
			50,
			100,
			500,
			1000,
			5000,
			10000,
			50000,
		};

		public static void Main(string[] args)
		{
			var input = ReadInput();

			var sw = new Stopwatch();
			foreach (var dataSize in TrainingDataSizes)
			{
				sw.Start();
				var featureData = input.Item1.SubMatrix(0, dataSize, 0, input.Item1.ColumnCount);
				var results = input.Item2.SubVector(0, dataSize);

				var theta = CalculateTheta(featureData, results);

				var successes = 0;
				var total = 0;
				for (var i = dataSize; i < input.Item1.RowCount; i++)
				{
					var testData = input.Item1.Row(i);
					var guessedType = Hypothesis(theta, testData) >= 0.5 ? 1 : 0;
					if (guessedType == (int)input.Item2[i])
					{
						successes++;
					}
					total++;
				}

				sw.Stop();

				Console.WriteLine($"Size: {dataSize}, Success rate: {1.0 * successes / total} Elapsed: {sw.ElapsedMilliseconds / 1000.0} s");
				sw.Reset();
			}

			Console.Read();
		}

		private static Vector<double> CalculateTheta(Matrix<double> featureData, Vector<double> results)
		{
			var theta = Vector<double>.Build.Random(featureData.ColumnCount);
			var cost = Cost(theta, featureData, results);
			double costDifference;
			do
			{
				theta = UpdateTheta(theta, featureData, results);
				var newCost = Cost(theta, featureData, results);
				costDifference = cost - newCost;
				cost = newCost;
			}
			while (costDifference > 0.0001);
			return theta;
		}

		private static Tuple<Matrix<double>, Vector<double>> ReadInput()
		{
			var file = File.ReadAllLines("Data.csv").Select(s => s.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)
																  .Select(double.Parse)
																  .ToArray())
													.OrderBy(f => Guid.NewGuid())
													.ToArray();
			var featureMatrix = Matrix<double>.Build.Dense(file.Length, file[0].Length);
			var outputVector = Vector<double>.Build.Dense(file.Length);
			for (var i = 0; i < file.Length; i++)
			{
				var data = file[i];
				var features = new[] { 1.0 }.Concat(data.Take(featureMatrix.ColumnCount - 1))
											.ToArray();

				featureMatrix.SetRow(i, features);

				outputVector[i] = data[featureMatrix.ColumnCount - 1] - 1;
				if (outputVector[i] != 0 && outputVector[i] != 1)
				{
					throw new ArgumentOutOfRangeException();
				}
			}

			var featureCount = featureMatrix.ColumnCount;
			var higherOrderMatrix = Matrix<double>.Build.Dense(featureMatrix.RowCount, 1 + 3 * featureCount * (featureCount - 1) / 2);
			higherOrderMatrix.SetSubMatrix(0, 0, featureMatrix);
			for (var i = 0; i < featureMatrix.RowCount; i++)
			{
				var counter = featureCount;
				for (var j = 1; j < featureCount; j++)
				{
					higherOrderMatrix[i, counter++] = Math.Pow(featureMatrix[i, j], 2);
					higherOrderMatrix[i, counter++] = Math.Pow(featureMatrix[i, j], 3);
				}

				for (var j = 1; j < featureCount; j++)
				{
					for (var k = 1; k < j; k++)
					{
						higherOrderMatrix[i, counter++] = featureMatrix[i, j] * featureMatrix[i, k];
						higherOrderMatrix[i, counter++] = Math.Pow(featureMatrix[i, j], 2) * featureMatrix[i, k];
						higherOrderMatrix[i, counter++] = featureMatrix[i, j] * Math.Pow(featureMatrix[i, k], 2);
					}
				}
			}

			foreach (var column in higherOrderMatrix.EnumerateColumnsIndexed())
			{
				if (column.Item1 == 0)
				{
					continue;
				}
				var min = column.Item2.Minimum();
				var max = column.Item2.Maximum();
				var mean = (max - min) / 2;
				for (var i = 0; i < column.Item2.Count; i++)
				{
					higherOrderMatrix[i, column.Item1] = max == min ? 0 : (higherOrderMatrix[i, column.Item1] - mean) / (max - min);
				}
			}

			return Tuple.Create(higherOrderMatrix, outputVector);
		}

		private static double Hypothesis(Vector<double> theta, Vector<double> features) => 1 / (1 + Math.Pow(Math.E, -theta.DotProduct(features)));

		private static Vector<double> UpdateTheta(Vector<double> theta, Matrix<double> trainingData, Vector<double> results)
		{
			var result = Vector<double>.Build.Dense(theta.Count, 0);
			for (var j = 0; j < theta.Count; j++)
			{
				result[j] = theta[j] - (Alpha / trainingData.RowCount) * trainingData.EnumerateRowsIndexed().Sum(r => (Hypothesis(theta, r.Item2) - results[r.Item1]) * r.Item2[j]);
			}
			return result;
		}

		private static double Cost(Vector<double> theta, Matrix<double> trainingData, Vector<double> results)
		{
			return (-1.0 / trainingData.RowCount) * (trainingData.EnumerateRowsIndexed()
																 .Sum(r => results[r.Item1] * Math.Log(Hypothesis(theta, r.Item2)) +
																		   (1 - results[r.Item1]) * Math.Log(1 - Hypothesis(theta, r.Item2))));
		}
	}
}