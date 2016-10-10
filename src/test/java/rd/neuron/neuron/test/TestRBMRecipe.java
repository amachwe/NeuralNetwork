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
		DataStreamer train = pb.getDataSet(unitLength,noOfUnits, 100000, 0.001f);
		DataStreamer test = pb.getDataSet(unitLength,noOfUnits, 100000, 0.10f);

		String recipe = "STOCHASTIC 96 24";

		List<LayerIf> network = RecipeNetworkBuilder.build(recipe);
		StochasticNetwork nw = new StochasticNetwork(network, Function.LOGISTIC, Function.LOGISTIC);
		for (int i = 0; i < epoch; i++) {
			for (FloatMatrix input : train) {
				nw.preTrain(input, 10, 0.02f);
			}
		}
		float avg = 0;
		int count = 0;
		for (FloatMatrix input : test) {
			FloatMatrix h = Propagate.up(input, network);

			FloatMatrix v = Propagate.down(h, network);
			System.out.println(input + "  " + v);
			avg+=PatternBuilder.score(v, input);
			count++;
		}
		
		System.out.println(avg/count);
	}
}