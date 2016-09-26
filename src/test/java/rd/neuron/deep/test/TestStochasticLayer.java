package rd.neuron.deep.test;

import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.neuron.neuron.StochasticLayer;

/**
 * Test Stochastic Layer for processing and training.
 * (Based on JBLAS Matrix Library).
 * @author azahar
 *
 */
public class TestStochasticLayer {

	@Test
	public void doTest() {
		FloatMatrix weights = FloatMatrix.rand(6, 3);
		FloatMatrix outputBias = FloatMatrix.rand(1,3);
		FloatMatrix inputBias = FloatMatrix.rand(1,6);
		StochasticLayer sl = new StochasticLayer(weights, outputBias, inputBias, new Random(123));

		FloatMatrix input = new FloatMatrix(1, 6);
		input.put(0, 0, 0.7f).put(0, 1, 0.5f).put(0, 2, 0.3f).put(0, 3, 0.1f).put(0, 4, 0.3f).put(0, 5, 0.5f);

		FloatMatrix result = new FloatMatrix(1,6);
		result.put(0, 0, 1f).put(0, 1, 0f).put(0, 2, 1f).put(0, 3, 1f).put(0, 4, 0f).put(0, 5, 1f);
		assertTrue(sl.stochasticLayer(input).equals(result));
		
		FloatMatrix output = new FloatMatrix(1,3);
		output.put(0, 0,1f).put(0, 1,1f).put(0, 2,1f);
		System.out.println(sl.oi(output));
		
		sl.train(input,10, 0.02f);

	}
}
