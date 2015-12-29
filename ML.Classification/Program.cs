using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace ML.Classification
{
	public class Program
	{
		private const double Alpha = 0.1;
		private const int TestSize = 100;

		private static readonly Random rand = new Random();
		private static readonly int[] TrainingDataSizes =
		{
			10,
			100,
			500,
			1000
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

				var thetaDict = new Dictionary<YeastType, Vector<double>>();
				foreach (var type in Enum.GetValues(typeof(YeastType)).OfType<YeastType>().Where(t => t != YeastType.Other))
				{
					var partialResults = Vector<double>.Build.Dense(results.Count);
					for (var i = 0; i < results.Count; i++)
					{
						partialResults[i] = (YeastType)(int)results[i] == type ? 1 : 0;
					}

					thetaDict[type] = CalculateTheta(featureData, partialResults);
				}

				var successLog = new List<Tuple<YeastType, bool>>();
				for (var i = 0; i < TestSize; i++)
				{
					var index = rand.Next(0, input.Item1.RowCount);
					var testData = input.Item1.Row(index);

					var type = YeastType.Other;
					var confidence = 0.0;
					foreach (var thetaKVP in thetaDict)
					{
						var theta = thetaKVP.Value;
						var result = Hypothesis(theta, testData);
						if (result > confidence)
						{
							type = thetaKVP.Key;
							confidence = result;
						}
					}

					var actualType = (YeastType)(int)input.Item2[index];
					successLog.Add(Tuple.Create(actualType, type == actualType));
				}

				sw.Stop();

				Console.WriteLine($"Size: {dataSize}, Elapsed: {sw.ElapsedMilliseconds / 1000.0} s");
				var successes = successLog.OrderBy(s => s.Item1)
										  .GroupBy(s => s.Item1)
										  .Select(g => new
										  {
											  Type = g.Key,
											  Successful = g.Count(s => s.Item2),
											  Total = g.Count()
										  });
				foreach (var success in successes)
				{
					Console.WriteLine($"Success rate for {success.Type}: {success.Successful} / {success.Total}");
				}

				Console.WriteLine($"Overall success rate: {successLog.Count(s => s.Item2)} / {successLog.Count}");
				Console.WriteLine("-----------------------------------");
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
			var file = File.ReadAllLines("Data.csv").Select(s => s.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries))
													.ToArray();
			var featureMatrix = Matrix<double>.Build.Dense(file.Length, file[0].Length);
			var outputVector = Vector<double>.Build.Dense(file.Length);
			for (var i = 0; i < file.Length; i++)
			{
				var data = file[i];
				var features = new[] { 1.0 }.Concat(data.Select(double.Parse)
														.Take(featureMatrix.ColumnCount - 1))
											.ToArray();


				featureMatrix.SetRow(i, features);

				YeastType result;
				if (!Enum.TryParse<YeastType>(data.Last(), out result))
				{
					result = YeastType.Other;
				}
				outputVector[i] = (int)result;
			}

			//foreach (var column in featureMatrix.EnumerateColumnsIndexed())
			//{
			//	if (column.Item1 == 0)
			//	{
			//		continue;
			//	}
			//	var min = column.Item2.Minimum();
			//	var max = column.Item2.Maximum();
			//	var mean = column.Item2.Average();
			//	for (var i = 0; i < column.Item2.Count; i++)
			//	{
			//		featureMatrix[i, column.Item1] = (featureMatrix[i, column.Item1] - mean) / (max - min);
			//	}
			//}

			var featureCount = featureMatrix.ColumnCount;
			var secondOrderMatrix = Matrix<double>.Build.Dense(featureMatrix.RowCount, 3 * featureCount);
			secondOrderMatrix.SetSubMatrix(0, 0, featureMatrix);
			for (var i = 0; i < featureMatrix.RowCount; i++)
			{
				for (var j = 0; j < featureCount; j++)
				{
					secondOrderMatrix[i, 1 * featureCount + j] = Math.Pow(featureMatrix[i, j], 2);
					secondOrderMatrix[i, 2 * featureCount + j] = Math.Pow(featureMatrix[i, j], 3);
				}
			}

			return Tuple.Create(secondOrderMatrix, outputVector);
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

	public enum YeastType
	{
		Other = 0,
		CYT = 1,
		NUC = 2,
		MIT = 3,
		ME3 = 4,
		ME2 = 5,
		ME1 = 6,
		EXC = 7,
		VAC = 8,
		POX = 9,
		ERL = 10,
	}
}