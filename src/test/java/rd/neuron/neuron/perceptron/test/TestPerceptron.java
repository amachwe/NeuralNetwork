package rd.neuron.neuron.perceptron.test;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.ConfusionMatrix;
import rd.data.DataStreamer;
import rd.neuron.neuron.perceptron.Perceptron;

public class TestPerceptron {

	private final int EPOCHS = 2000;
/**
 * AND Gate test 
 */
	@Test
	public void doANDTest() {
		ConfusionMatrix cm = new ConfusionMatrix();
		DataStreamer input = new DataStreamer(2, 1);
		// AND data set

		input.add(new float[] { 0f, 0.0f }, -1f);
		input.add(new float[] { 1f, 0.0f }, -1f);
		input.add(new float[] { 0f, 1.0f }, -1f);
		input.add(new float[] { 1f, 1.0f }, 1f);

		float[] LEARNING_RATE = new float[] { 0.5f, 0.2f, 0.09f, 0.08f, 0.07f, 0.06f, 0.05f, 0.02f };
		for (int j = 0; j < LEARNING_RATE.length; j++) {
			Perceptron p = new Perceptron(2);
			for (int i = 0; i < EPOCHS; i++) {

				p.train(input, LEARNING_RATE[j]);

			}
			for (FloatMatrix item : input) {

				int actual = (int) input.getOutput(item).sum();
				int calculated = p.predict(item.toArray());
				if (actual > 0 && calculated > 0) {
					cm.incTP();
				} else if (actual < 0 && calculated < 0) {
					cm.incTN();
				} else if (actual < 0 && calculated > 0) {
					cm.incFP();
				} else if (actual > 0 && calculated < 0) {
					cm.incFN();
				}
			}
			System.out.println("AND: "+LEARNING_RATE[j] + "," + cm);

		}
	}
	
	/**
	 * 
	 * XOR Gate should give poor performance because single layer perceptrons cannot approximate the XOR function
	 */
	@Test
	public void doXORTest() {
		ConfusionMatrix cm = new ConfusionMatrix();
		DataStreamer input = new DataStreamer(2, 1);
		// XOR data set

		input.add(new float[] { 0f, 0.0f }, 1f);
		input.add(new float[] { 1f, 0.0f }, -1f);
		input.add(new float[] { 0f, 1.0f }, -1f);
		input.add(new float[] { 1f, 1.0f }, 1f);

		float[] LEARNING_RATE = new float[] { 0.5f, 0.2f, 0.09f, 0.08f, 0.07f, 0.06f, 0.05f, 0.02f };
		for (int j = 0; j < LEARNING_RATE.length; j++) {
			Perceptron p = new Perceptron(2);
			for (int i = 0; i < EPOCHS; i++) {

				p.train(input, LEARNING_RATE[j]);

			}
			for (FloatMatrix item : input) {

				int actual = (int) input.getOutput(item).sum();
				int calculated = p.predict(item.toArray());
				if (actual > 0 && calculated > 0) {
					cm.incTP();
				} else if (actual < 0 && calculated < 0) {
					cm.incTN();
				} else if (actual < 0 && calculated > 0) {
					cm.incFP();
				} else if (actual > 0 && calculated < 0) {
					cm.incFN();
				}
			}
			System.out.println("XOR: "+LEARNING_RATE[j] + "," + cm);

		}
	}
}
