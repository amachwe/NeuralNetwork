package rd.neuron.neuron.test;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.TimedDistributionStructure;
import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.LayerIf;
import rd.neuron.neuron.LayerIf.LayerType;
import rd.neuron.neuron.Propagate;
import rd.neuron.neuron.RecipeNetworkBuilder;
import rd.neuron.neuron.StochasticNetwork;
import rd.neuron.neuron.TrainNetwork;

public class TestRecipeBuilder {

	private static final TimedDistributionStructure<String, String> tds = new TimedDistributionStructure<>(4500, 20, 2);

	public static FloatMatrix[] createTest(int count, int rows, int cols, float pattern[][]) {
		FloatMatrix[] x = new FloatMatrix[count];
		for (int i = 0; i < count; i++) {
			x[i] = new FloatMatrix(rows, cols);
			for (int j = 0; j < rows; j++) {
				for (int k = 0; k < cols; k++) {
					x[i].put(j, k, Math.random() < pattern[j][k] ? 1 : 0);
				}
			}
		}

		return x;
	}

	@Test
	public void doTest() {
		String recipe = "STOCHASTIC 12 8\nRANDOM 8 2";

		List<LayerIf> network = RecipeNetworkBuilder.build(recipe);
		StochasticNetwork nw = new StochasticNetwork(network, Function.LOGISTIC, Function.LOGISTIC);

		FloatMatrix input[] = createTest(20000, 12, 1, new float[][] { { 0.5f }, { 0.5f }, { 0.5f }, { 0.5f }, { 0.5f },
				{ 0.5f }, { 0.5f }, { 0.5f }, { 0.5f }, { 0.5f }, { 0.5f }, { 0.5f } });

		FloatMatrix train[] = Arrays.copyOfRange(input, 0, input.length / 2);
		FloatMatrix test[] = Arrays.copyOfRange(input, (input.length / 2) + 1, input.length);

		// for (FloatMatrix in : test) {
		// FloatMatrix res = null;
		// System.out.println(in + " ---- " + (res = result(network, in)));
		// addToTDS(in.elementsAsList().toString().replace(",", ""),
		// res.elementsAsList().toString().replace(",", ""));
		// }

		System.out.println("Pre-training");

		for (FloatMatrix in : train) {
			nw.preTrain(in);
		}

		// tds.nextTimeslice();
		// for (FloatMatrix in : test) {
		// FloatMatrix res = null;
		// System.out.println(in + " ---- " + (res = result(network, in)));
		// addToTDS(in.elementsAsList().toString().replace(",", ""),
		// res.elementsAsList().toString().replace(",", ""));
		// }

		for (int epoch = 0; epoch < 1000; epoch++) {
			for (FloatMatrix in : input) {

				// Simple Back Prop Training of Output Layer
				FloatMatrix actualOutput = Propagate.up(in, network);
				FloatMatrix expectedOutput = expectedOutput(in);

				nw.fineTuneOutputLayer(expectedOutput, actualOutput, in);

			}
			if (epoch % 100 == 0)
				System.out.println(epoch);
		}
		float countPos = 0, countNeg = 0;

		for (FloatMatrix t : test) {
			FloatMatrix actualOutput = Propagate.up(t, network);
			FloatMatrix expectedOutput = expectedOutput(t);

			for (int i = 0; i < expectedOutput.rows; i++) {
				for (int j = 0; j < expectedOutput.columns; j++) {
					if ((Math.abs(actualOutput.get(i, j) - expectedOutput.get(i, j)) < 0.1)) {
						countPos++;
					} else {
						countNeg++;
						System.out.println("Class:" + t.sum() + "\n" + actualOutput + "    " + expectedOutput + "\n\n");
					}
				}
			}
		}

		System.out.println(countPos + "  " + countNeg + "   " + (countPos * 100 / (countPos + countNeg)));
		System.out.println(network);
		tds.writeToFile(new File("test_1.csv"), 0);
		tds.writeToFile(new File("test_2.csv"), 1);
	}

	private FloatMatrix expectedOutput(FloatMatrix in) {
		// Output: If Majority 1 then 0,1, if majority 0 then 1,0, if all 1 then
		// 1,1, if all 0 then 0,0
		FloatMatrix outputExpected = new FloatMatrix(2, 1);
		float oneCount = 0;
		for (float f : in.data) {
			if (f > 0) {
				oneCount++;
			}
		}
		if (oneCount == 0 || oneCount == in.length || oneCount == in.length / 2) {
			oneCount = oneCount / in.length;
			outputExpected.put(1, 0, oneCount);
			outputExpected.put(0, 0, oneCount);
		} else {
			if (oneCount > in.length / 2) {
				outputExpected.put(1, 0, 1);
				outputExpected.put(0, 0, 0);
			} else {
				outputExpected.put(1, 0, 0);
				outputExpected.put(0, 0, 1);
			}
		}
		return outputExpected;
	}

	private void addToTDS(String in, String out) {
		try {
			tds.add(in, out);
		} catch (Exception e) {

			e.printStackTrace(System.err);
		}
	}

	private FloatMatrix result(List<LayerIf> network, FloatMatrix in) {
		FloatMatrix result = null;
		for (LayerIf layer : network) {

			if (result == null) {

				result = layer.io(in);

			} else {

				result = layer.io(result);
			}
		}

		return result;
	}
}
