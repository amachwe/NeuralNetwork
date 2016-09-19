package rd.neuron.neuron;

import java.util.Random;

import org.jblas.FloatMatrix;

public class StochasticLayer extends Layer {

	private final Random rnd;

	public StochasticLayer(FloatMatrix weights, FloatMatrix bias, FloatMatrix inputBias, Random rnd) {
		super(weights, bias, inputBias, Function.LOGISTIC);

		this.rnd = rnd;
	}

	public FloatMatrix stochasticLayer(FloatMatrix input) {
		FloatMatrix output = new FloatMatrix(input.rows, input.columns);
		
		for (int i = 0; i < input.columns; i++) {
			for (int j = 0; j < input.rows; j++) {
				output.put( j,i, input.get(j,i) > rnd.nextFloat() ? 1f : 0f);
			}
		}
		return output;
	}

	public void train(FloatMatrix input, int batchSize, int iter, float learningRate) {

		if (iter <= 0) {
			iter = 1;
		}

		FloatMatrix vk = null, hk = null,hk0=null, hm = null, hm0 = null;
		// p("\n **************** INPUT",input);
		hk0 = stochasticLayer(hm0 = io(input));
		// p("HK",hk);
		// p("HM0",hm0);
		for (int i = 0; i < iter; i++) {

			if (i == 0) {

				vk = stochasticLayer(oi(hk0));
				// p("VK",vk);
				hk = stochasticLayer(hm = io(vk));
			} else {
				vk = stochasticLayer(oi(hk));
				// p("VK",vk);
				hk = stochasticLayer(hm = io(vk));
			}
			// p("HM",hm);
			// p("HK",hk);

		}
		updateWeights(learningRate, batchSize, hm0, hm, input, vk);
		updateHiddenBias(learningRate, batchSize, hk0, hm);
		updateVisibleBias(learningRate, batchSize, input, vk);

	}

	private void updateHiddenBias(float learningRate, int batchSize, FloatMatrix initialHiddenSample,
			FloatMatrix currentHiddenMean) {

		bias = bias.add((initialHiddenSample.sub(currentHiddenMean)).mul(learningRate / batchSize));
	}

	private void updateVisibleBias(float learningRate, int batchSize, FloatMatrix input,
			FloatMatrix currentVisibleSample) {

		inputBias = inputBias.add((input.sub(currentVisibleSample)).mul(learningRate / batchSize));
	}

	public static void p(String s, FloatMatrix v) {
		System.out.println(s + " : " + v.rows + "x" + v.columns);
		System.out.println(v);
	}

	private void updateWeights(float learningRate, int batchSize, FloatMatrix initialHiddenMean,
			FloatMatrix currentHiddenMean, FloatMatrix input, FloatMatrix visibleSample) {
		// p("IHM", initialHiddenMean);
		// p("IN", input);
		// p("WTS",weights);
		FloatMatrix init = input.mmul(initialHiddenMean.transpose());
		FloatMatrix fin = visibleSample.mmul(currentHiddenMean.transpose());
		// p("INIT", init);
		// p("FIN", fin);
		// p("WTS",weights);
		FloatMatrix delta = init.sub(fin);
		weights = weights.add(delta.mul(learningRate / batchSize));

	}

}
