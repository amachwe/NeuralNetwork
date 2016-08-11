package rd.neuron.neuron.test;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import rd.neuron.neuron.Hopfield;

/**
 * Hopfield network tests
 * @author azahar
 *
 */
public class TestHopfield {

	@Test
	public void doHebbTrainingTest() {
		float[][] patterns = new float[][] { { 1, 0, 0, 1 }, { 0, 1, 1, 0 } };
		Hopfield hf = new Hopfield(patterns[0].length);
		hf.addPatternHebbian(patterns[0], patterns.length);
		hf.addPatternHebbian(patterns[1], patterns.length);
		assertTrue(hf.getWeights()[0][3] == 0.5f);
		assertTrue(hf.getWeights()[1][2] == 0.5f);
		assertTrue(hf.getWeights()[2][1] == 0.5f);
		assertTrue(hf.getWeights()[3][0] == 0.5f);
		assertTrue(hf.getWeights()[0][2] == 0.0f);
		assertTrue(hf.getWeights()[3][2] == 0.0f);
		assertTrue(hf.getWeights()[2][0] == 0.0f);
		assertTrue(hf.getWeights()[2][3] == 0.0f);

	}

	/**
	 * Pattern test
	 */
	@Test
	public void doTrainingTest() {
		float[][] patterns = new float[][] { { 1, 0, 0, 1 }, { 0, 1, 1, 0 } };
		Hopfield hf = new Hopfield(patterns[0].length);

		hf.addPatternHebbian(patterns[0], patterns.length);
		hf.addPatternHebbian(patterns[1], patterns.length);

		for (int i = 0; i < patterns[0].length; i++) {
			assertTrue(hf.getStates(new float[] { 1, 0, 0, 1 })[i] == patterns[0][i]);
		}

		assertTrue(hf.getStates(new float[] { 0, 1, 0, 0 })[2] == patterns[1][2]);
		assertTrue(hf.getStates(new float[] { 0, 1, 0, 0 })[0] == patterns[1][0]);

	}

}
