package rd.neuron.deep.test;

import java.util.Arrays;
import java.util.Random;

import org.junit.Test;

import rd.neuron.deep.Layer;
import rd.neuron.deep.NeuralElement;
import rd.neuron.deep.SigmoidBinomialSamplingLayer;
import rd.neuron.deep.training.ContrastiveDivergence;

/**
 * Test Layer Neural Element with Contrastive Divergence
 * @author azahar
 *
 */
public class TestLayer {

	Random rnd = new Random(1234l);

	@Test
	public void doTest() {
		int[][] inputs = { { 1, 0, 1 }, { 1, 1, 1 }, { 0, 0, 0 } };
		NeuralElement l = new SigmoidBinomialSamplingLayer(
				new Layer(new double[] { 1d, 1d, 1d }, new double[] { 1d, 1d }), rnd);
		l = ContrastiveDivergence.train(inputs, 100, 0.09f, l);
		
		for(int ex = 0;ex<inputs.length;ex++)
		{
			System.out.println(Arrays.toString(inputs[ex])+"    -    "+Arrays.toString(l.reconstruct(inputs[ex])));
		}
	}
}
