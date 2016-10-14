package rd.neuron.neuron.test;

import java.util.List;
import java.util.Random;

import org.jblas.FloatMatrix;

import rd.data.DataStreamer;
import rd.data.PatternBuilder;
import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.LayerIf;
import rd.neuron.neuron.Propagate;
import rd.neuron.neuron.RecipeNetworkBuilder;
import rd.neuron.neuron.StochasticNetwork;

public class TestRBMRecipe {

	private static PatternBuilder pb = new PatternBuilder(new Random(123));

	public static void main(String... args) {
		int epoch = 10000;
		int noOfUnits = 24;
		int unitLength = 4;
		int count = 100000;
		DataStreamer train = pb.getDataSet(unitLength, noOfUnits, count, 0.001f);
		DataStreamer test = pb.getDataSet(unitLength, noOfUnits, count, 0.10f);

		String recipe = "STOCHASTIC 96 24";

		List<LayerIf> network = RecipeNetworkBuilder.build(recipe);
		StochasticNetwork nw = new StochasticNetwork(network, Function.LOGISTIC, Function.LOGISTIC);
		for (int i = 0; i < epoch; i++) {

			FloatMatrix dataSet[] = new FloatMatrix[(int) (0.1 * count)];
			int c = 0;
			while (c < dataSet.length) {
				for (FloatMatrix in : train) {
					if (c < dataSet.length) {
						if (Math.random() < 0.1) {
							dataSet[c++] = in;
						}
					} else {
						break;
					}
				}
			}
			nw.preTrain(dataSet, 10, 0.02f);

		}
		float avg = 0;
		int _count = 0;
		for (FloatMatrix input : test) {
			FloatMatrix h = Propagate.up(input, network);

			FloatMatrix v = Propagate.down(h, network);
			System.out.println(input + "  " + v);
			avg += PatternBuilder.score(v, input);
			_count++;
		}

		System.out.println(avg / _count);
	}
}