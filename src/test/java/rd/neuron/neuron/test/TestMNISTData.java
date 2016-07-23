package rd.neuron.neuron.test;

import java.io.IOException;

import org.canova.image.mnist.MnistLabelFile;
import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.ConsoleDataWriter;
import rd.data.DataStreamer;
import rd.data.MnistImageFile;
import rd.neuron.neuron.FullyRandomLayerBuilder;
import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.SimpleNetwork;
import rd.neuron.neuron.SimpleNetworkPerformanceEvaluator;

public class TestMNISTData {

	private final int EPOCHS = 1000000;
	private final boolean SGD = true;
	private final float LEARNING_RATE = 0.01f;

	@Test
	public void doMNISTTest() throws IOException {
		DataStreamer streamer = new DataStreamer(784, 10);

		MnistImageFile digitImage = new MnistImageFile("d:\\ml stats\\mnist\\t10k-images.idx3-ubyte", "r");
		MnistLabelFile labelData = new MnistLabelFile("d:\\ml stats\\mnist\\t10k-labels.idx1-ubyte", "r");
		final float max = 255f;
		while (digitImage.getCurrentIndex() <= digitImage.getCount()) {
			digitImage.nextImage();

			labelData.next();

			int lbl = labelData.readLabel();
			int img[][] = digitImage.readImage();
			float[] imgInput = new float[img.length*img[0].length];
			int k=0;
		
			for(int i=0;i<img.length;i++)
			{
				for(int j=0;j<img[0].length;j++)
				{
				
					imgInput[k++] = (float)(img[i][j]/max);
				}
				
				
			}
		
			
			streamer.add(imgInput, digitToOneHotEncoding(lbl));

		}
		
		

		SimpleNetwork network = new SimpleNetwork(new FullyRandomLayerBuilder(0.5f,1f), Function.LOGISTIC, 784, 15, 10);
		
		SimpleNetworkPerformanceEvaluator snpm = new SimpleNetworkPerformanceEvaluator(new ConsoleDataWriter());
		// Back Prop
		for (int i = 0; i < EPOCHS; i++) {
			System.out.println(i*100f/EPOCHS);
			if (!SGD) {
				for (FloatMatrix item : streamer) {

					FloatMatrix outputLayerNewWts[] = network.trainOutputLayer(LEARNING_RATE, streamer.getOutput(item),
							network.io(item));
					FloatMatrix hiddenLayerNewWts[] = network.trainHiddenLayer(0, LEARNING_RATE,
							streamer.getOutput(item), network.io(item), item);

					network.setOutputWeights(outputLayerNewWts[0]);
					network.setOutputBias(outputLayerNewWts[1]);
					network.setWeights(0, hiddenLayerNewWts[0]);
					network.setBias(0, hiddenLayerNewWts[1]);
				}
			} else {
				FloatMatrix item = streamer.getRandom();

				FloatMatrix outputLayerNewWts[] = network.trainOutputLayer(LEARNING_RATE, streamer.getOutput(item),
						network.io(item));
				FloatMatrix hiddenLayerNewWts[] = network.trainHiddenLayer(0, LEARNING_RATE, streamer.getOutput(item),
						network.io(item), item);

				network.setOutputWeights(outputLayerNewWts[0]);
				network.setOutputBias(outputLayerNewWts[1]);
				network.setWeights(0, hiddenLayerNewWts[0]);
				network.setBias(0, hiddenLayerNewWts[1]);

			}
			//snpm.evaluate(network, streamer);
		}
		
		for (FloatMatrix item : streamer) {
			System.out.println(" > Actual: " + network.io(item) + "  > Expected: " + streamer.getOutput(item));
		}
	}

	private float[] digitToOneHotEncoding(int digit) {
		switch (digit) {
		case 0:
			return new float[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		case 1:
			return new float[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
		case 2:
			return new float[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
		case 3:
			return new float[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
		case 4:
			return new float[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
		case 5:
			return new float[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
		case 6:
			return new float[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
		case 7:
			return new float[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };
		case 8:
			return new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
		case 9:
			return new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
		default:
			return new float[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		}
	}
}
