package rd.neuron.neuron.test;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.imageio.ImageIO;

import org.jblas.FloatMatrix;

import rd.data.DataStreamer;
import rd.data.MnistToDataStreamer;
import rd.data.PatternBuilder;
import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.LayerIf;
import rd.neuron.neuron.Propagate;
import rd.neuron.neuron.RecipeNetworkBuilder;
import rd.neuron.neuron.StochasticNetwork;

public class TestRBMMNISTRecipe {

	/**
	 * Files for Labels/Images - Training and Testing - change to run tests
	 */
	private static final String D_ML_STATS_MNIST_TRAIN_LABELS = "data\\train-labels.idx1-ubyte";
	private static final String D_ML_STATS_MNIST_TRAIN_IMAGES = "data\\train-images.idx3-ubyte";
	private static final String D_ML_STATS_MNIST_T10K_LABELS = "data\\t10k-labels.idx1-ubyte";
	private static final String D_ML_STATS_MNIST_T10K_IMAGES = "data\\t10k-images.idx3-ubyte";

	private static PatternBuilder pb = new PatternBuilder(new Random(123));

	public static void main(String... args) throws FileNotFoundException, IOException {

		ThreadedImageWriter tiw = new ThreadedImageWriter(4);
		int epoch = 1000;

		DataStreamer train = MnistToDataStreamer.createStreamer(D_ML_STATS_MNIST_T10K_IMAGES,
				D_ML_STATS_MNIST_T10K_LABELS);
		System.out.println("Training Streamer Ready!");

		DataStreamer test = MnistToDataStreamer.createStreamer(D_ML_STATS_MNIST_TRAIN_IMAGES,
				D_ML_STATS_MNIST_TRAIN_LABELS);
		System.out.println("Testing Streamer Ready!");

		int width = 10;
		String recipe = "STOCHASTIC 784 " + (int) (width * width);

		List<LayerIf> network = RecipeNetworkBuilder.build(recipe);
		StochasticNetwork nw = new StochasticNetwork(network, Function.LOGISTIC, Function.LOGISTIC);
		for (int i = 0; i < epoch; i++) {
			for (FloatMatrix input : train) {
				if (Math.random() < 0.001) {
					nw.preTrain(input, 10, 0.02f);
				}
			}
			if (epoch % 100 == 0) {
				System.out.println(i * 100f / epoch);
			}
		}
		float avg = 0;
		int count = 0;
		int maxI = 10;
		for (FloatMatrix input : test) {

			FloatMatrix h = Propagate.up(input, network);

			FloatMatrix v = Propagate.down(h, network);

			avg += PatternBuilder.score(v, input, 0.1f);
			if (Math.random() < 0.01 && maxI > 0) {
				tiw.writeImage(input, 28, 28, "input." + maxI + ".png");
				tiw.writeImage(h, width, width, "hidden." + maxI + ".png");
				tiw.writeImage(v, 28, 28, "visible." + maxI + ".png");
				maxI--;
			}
			count++;
		}

		for (int i = 0; i < 10; i++) {
			FloatMatrix rand = FloatMatrix.zeros(width * width,1);
			for (int j = 0; j < width * width; j++) {
				if (Math.random() > 0.5) {
					rand.put(j,0, 1f);
				} 
			}
			System.out.println(Propagate.down(rand, network));
			tiw.writeImage(Propagate.down(rand, network), width, width, "random." + i + ".png");
		}

		System.out.println(avg / count);
		tiw.shutdown();

	}

}

class ThreadedImageWriter {
	private final ExecutorService es;

	public ThreadedImageWriter(int threads) {
		es = Executors.newFixedThreadPool(threads);
	}

	public void writeImage(FloatMatrix fm, int width, int height, String filename) {

		es.execute(new Task(fm, width, height, filename));
	}

	public void shutdown() {
		es.shutdown();
	}

}

class Task implements Runnable {
	private final FloatMatrix fm;
	private final int width, height;
	private final String filename;

	public Task(FloatMatrix fm, int width, int height, String filename) {
		this.fm = fm;
		this.width = width;
		this.height = height;
		this.filename = filename;
	}

	@Override
	public void run() {
		System.out.println("Writing: " + filename);
		BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		int c = 0;
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {

				bi.setRGB(j, i, (int) (255 * fm.get(c++)));
			}
		}

		try {
			ImageIO.write(bi, "png", new File(filename));

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			System.out.println("End: " + filename);
		}
	}
}