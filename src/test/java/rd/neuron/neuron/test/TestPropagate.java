package rd.neuron.neuron.test;

import java.io.File;
import java.util.List;
import java.util.Random;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.TimedDistributionStructure;
import rd.neuron.neuron.LayerIf;
import rd.neuron.neuron.Propagate;
import rd.neuron.neuron.RecipeNetworkBuilder;

public class TestPropagate {

	@Test
	public void doPropTest() throws Exception {
		Random rnd = new Random(123);
		FloatMatrix input[] = createTestUniform(32, 4, 1,rnd);// createTest(20, 4,
															// 1, new float[][]
															// { { 0.5f }, {
															// 0.5f }, { 0.5f },
															// { 0.5f } }, rnd);
		TimedDistributionStructure<String, String> tds = new TimedDistributionStructure<>(16, 4, 1);
		String recipe = "STOCHASTIC 4 2";

		List<LayerIf> network = RecipeNetworkBuilder.build(recipe);
		LayerIf l = network.get(0);
		int rows = l.getWeights().rows;
		int cols = l.getWeights().columns;
		FloatMatrix newWts = new FloatMatrix(rows, cols);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				newWts.put(i, j, rnd.nextFloat());
			}
		}

		l.setAllWeights(newWts);
		for (FloatMatrix in : input) {
			System.out.println(">>" + in);
			List<FloatMatrix> list = Propagate.upDown(in, l, 10);
			String prevItem = null;
			for (FloatMatrix f : list) {
				String item = f.elementsAsList().toString().replace(",", " ").replace(".0", "");
				if(prevItem!=null)
				{
				System.out.println(prevItem+"        "+item);
				}
				if (prevItem != null) {
					if (prevItem.length() < item.length()) {
						tds.add(item, prevItem);
					} else {
						tds.add(prevItem, item);
					}
				}
				prevItem = item;
			}

			System.out.println("\n\n");

		}

		tds.writeToFile(new File("gs_ex.csv"), 0);

	}

	public static FloatMatrix[] createTest(int count, int rows, int cols, float pattern[][], Random rnd) {
		if (rnd == null) {
			rnd = new Random(System.currentTimeMillis());
		}
		FloatMatrix[] x = new FloatMatrix[count];
		for (int i = 0; i < count; i++) {
			x[i] = new FloatMatrix(rows, cols);
			for (int j = 0; j < rows; j++) {
				for (int k = 0; k < cols; k++) {
					x[i].put(j, k, rnd.nextFloat() < pattern[j][k] ? 1 : 0);
				}
			}
		}

		return x;
	}

	public static FloatMatrix[] createTestUniform(int count, int rows, int cols,Random rnd) {
		if (rnd == null) {
			rnd = new Random(System.currentTimeMillis());
		}
		FloatMatrix[] x = new FloatMatrix[count];
		for (int i = 0; i < count; i++) {
			x[i] = new FloatMatrix(rows, cols);
			String str = Integer.toBinaryString(i);

			int c = 0;
			int[] data = new int[rows];
			if (str.length() < rows) {
				// pad with 0.
				while (c + str.length() < rows) {
					data[c++] = 0;
				}
			}
			if (str.length() > rows) {
				
				for (int s = 0; s < rows; s++) {
					data[s] =  rnd.nextFloat() >0.5 ? 1:0;
				}
			
			} else {

				for (int s = c; s < str.length(); s++) {
					data[s] = Integer.parseInt("" + str.charAt(s));
				}
			}
			for (int j = 0; j < cols; j++) {
				for (int k = 0; k < rows; k++) {

					x[i].put(k, j, data[k]);
				}
			}
		}

		return x;
	}

}
