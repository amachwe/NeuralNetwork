package rd.neuron.neuron.test;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.TimedDistributionStructure;
import rd.neuron.neuron.LayerIf;
import rd.neuron.neuron.LayerIf.LayerType;
import rd.neuron.neuron.Propagate;
import rd.neuron.neuron.RecipeNetworkBuilder;

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
		String recipe = "STOCHASTIC 12 6\nSTOCHASTIC 6 6\nSTOCHASTIC 6 4";

		List<LayerIf> network = RecipeNetworkBuilder.build(recipe);

		FloatMatrix input[] = createTest(1000, 12, 1, new float[][] { { 0.2f }, { 0.5f }, { 0.3f }, { 0.6f }, { 0.5f },
				{ 0.7f }, { 0.7f }, { 0.7f }, { 0.5f }, { 0.3f }, { 0.4f }, { 0.8f } });
		FloatMatrix train[] = Arrays.copyOfRange(input, 0, input.length / 2);
		FloatMatrix test[] = Arrays.copyOfRange(input, (input.length / 2) + 1, input.length);

	

		for (FloatMatrix in : test) {
			FloatMatrix res = null;
			System.out.println(in + " ---- " + (res = result(network, in)));
			addToTDS(in.elementsAsList().toString().replace(",", ""), res.elementsAsList().toString().replace(",", ""));
		}

		System.out.println("Pre-training");

		for (LayerIf layer : network) {

			if (layer.getLayerType() == LayerType.FIRST_HIDDEN) {
				for (FloatMatrix in : train) {
					layer.train(in, 10, 0.02f);

				}
				

			} else {
				if (layer.getLayerType() != LayerType.OUTPUT) {

					for (FloatMatrix in : train) {

						FloatMatrix _result = null;

						_result = Propagate.up(in, network,layer.getLayerIndex());

						layer.train(_result, 10, 0.02f);

					}
				}

			}
		}

		tds.nextTimeslice();
		for (FloatMatrix in : test) {
			FloatMatrix res = null;
			System.out.println(in + " ---- " + (res = result(network, in)));
			addToTDS(in.elementsAsList().toString().replace(",", ""), res.elementsAsList().toString().replace(",", ""));
		}

		tds.writeToFile(new File("test_1.csv"), 0);
		tds.writeToFile(new File("test_2.csv"), 1);
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
