package rd.neuron.neuron.test;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.imageio.ImageIO;

import org.jblas.FloatMatrix;

import com.aliasi.util.Files;

import rd.data.DataStreamer;
import rd.data.MnistToDataStreamer;
import rd.data.PatternBuilder;
import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.LayerIf;
import rd.neuron.neuron.Propagate;
import rd.neuron.neuron.RecipeNetworkBuilder;
import rd.neuron.neuron.StochasticLayer;
import rd.neuron.neuron.StochasticNetwork;

public class TestRBMMNISTRecipe {

	private static final String fileName = "network2.out.nw";
	private static final boolean loadFromFile = true;

	private static final Random rnd = new Random(123);
	/**
	 * Files for Labels/Images - Training and Testing - change to run tests
	 */
	private static final String D_ML_STATS_MNIST_TRAIN_LABELS = "data\\train-labels.idx1-ubyte";
	private static final String D_ML_STATS_MNIST_TRAIN_IMAGES = "data\\train-images.idx3-ubyte";
	private static final String D_ML_STATS_MNIST_T10K_LABELS = "data\\t10k-labels.idx1-ubyte";
	private static final String D_ML_STATS_MNIST_T10K_IMAGES = "data\\t10k-images.idx3-ubyte";

	public static void main(String... args) throws FileNotFoundException, IOException {

		ThreadedImageWriter tiw = new ThreadedImageWriter(4);
		int epoch = 10000;

		DataStreamer train = MnistToDataStreamer.createStreamer(D_ML_STATS_MNIST_T10K_IMAGES,
				D_ML_STATS_MNIST_T10K_LABELS);
		System.out.println("Training Streamer Ready!");

		DataStreamer test = MnistToDataStreamer.createStreamer(D_ML_STATS_MNIST_TRAIN_IMAGES,
				D_ML_STATS_MNIST_TRAIN_LABELS);
		System.out.println("Testing Streamer Ready!");

		List<LayerIf> network;
		int width = 10;
		int lengthHidden = width * width;
		if (!(new File(fileName)).exists() || !loadFromFile) {

			String recipe = "STOCHASTIC 784 " + lengthHidden + "\nSTOCHASTIC " + lengthHidden + " 100\nSTOCHASTIC "
					+ lengthHidden + " " + lengthHidden + "\nSTOCHASTIC " + lengthHidden + " " + lengthHidden;

			network = RecipeNetworkBuilder.build(recipe);
			StochasticNetwork nw = new StochasticNetwork(network, Function.LOGISTIC, Function.LOGISTIC);
			List<FloatMatrix> dataSet = new ArrayList<>();
			for (int i = 0; i < epoch; i++) {
				dataSet.clear();
				for (FloatMatrix input : train) {
					if (Math.random() < 0.001) {
						dataSet.add(input);
					}
				}
				nw.preTrain(dataSet, 10, 0.02f);

				if (epoch % 100 == 0) {
					System.out.println(i * 100f / epoch);
				}
			}

			StochasticNetwork.save(fileName, network);
		} else {
			network = StochasticNetwork.load(fileName);
		}
		float avg = 0;
		int count = 0;
		int maxI = 20;
		int recordLen = 4;
		FloatMatrix fm[][] = new FloatMatrix[maxI][recordLen + 2];
		FloatMatrix[] record = new FloatMatrix[recordLen];
		for (FloatMatrix input : test) {

			FloatMatrix h = input;
			int lcount = 0;
			for (LayerIf l : network) {
				h = StochasticLayer.stochasticLayer(Propagate.upOne(h, l), rnd);
				record[lcount++] = h;
			}

			FloatMatrix v = StochasticLayer.stochasticLayer(Propagate.down(h, network), rnd);

			avg += PatternBuilder.score(v, input, 0.1f);
			if (Math.random() < 0.01 && maxI > 0) {
				maxI--;
				fm[maxI][0] = input;
				fm[maxI][1] = v;
				for (int cc = 0; cc < record.length; cc++) {
					fm[maxI][cc + 2] = record[cc];
				}

			}
			count++;
		}

		tiw.writeImage(fm, 28, 28, "combine.png");

		int rh = 0, rw = 0;

		int maxH = 20, maxW = 20;
		int ranCon = maxH * maxW;
		int maxSample= 50;
		FloatMatrix[][] randomGen = new FloatMatrix[maxH][maxW];
		for (int i = 0; i < ranCon; i++) {
			FloatMatrix rand = FloatMatrix.rand(lengthHidden, 1);
			for (int j = 0; j < lengthHidden; j++) {
				if (Math.random() > 0.5) {
					rand.put(j, 0, 1f);
				} else {
					rand.put(j, 0, 0f);
				}
			}

			FloatMatrix random = rand;
			
			for (int cc = 0; cc < maxSample; cc++) {
				random = StochasticLayer.stochasticLayer(Propagate.down(random, network), rnd);

				random = StochasticLayer.stochasticLayer(Propagate.up(random, network), rnd);

			}

			random = StochasticLayer.stochasticLayer(Propagate.down(random, network), rnd);

			randomGen[rw++][rh] = random;
			if (rh >= maxH) {
				rh = 0;
			}
			if (rw >= maxW) {
				rw = 0;
				rh++;
			}

		}

		tiw.writeImage(randomGen, 28, 28, "random"+(maxSample+1)+"."+ranCon+".png");

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

	public void writeImage(FloatMatrix fm[][], int width, int height, String filename) {

		es.execute(new Task(fm, width, height, filename));
	}

	public void shutdown() {
		es.shutdown();
	}

}

class Task implements Runnable {
	private final FloatMatrix fm[][];
	private final int width, height;
	private final String filename;

	public Task(FloatMatrix fm, int width, int height, String filename) {
		this.fm = new FloatMatrix[][] { { fm } };
		this.width = width;
		this.height = height;
		this.filename = filename;
	}

	public Task(FloatMatrix fm[][], int width, int height, String filename) {
		this.fm = fm;
		this.width = width;
		this.height = height;
		this.filename = filename;
	}

	@Override
	public void run() {

		System.out.println("Writing: " + filename);
		int actualWidth = width * fm[0].length;
		int actualHeight = height * fm.length;

		BufferedImage bi = new BufferedImage(actualWidth, actualHeight, BufferedImage.TYPE_INT_RGB);
		for (int w = 0; w < fm.length; w++) {
			for (int h = 0; h < fm[0].length; h++) {
				int c = 0;
				int len = (int) Math.sqrt(fm[w][h].length);
				for (int i = 0; i < width; i++) {
					for (int j = 0; j < height; j++) {
						if (c >= fm[w][h].length) {
							bi.setRGB(j + (h * height), i + (w * width), 0);
						} else if (j <= len && i <= len) {
							bi.setRGB(j + (h * height), i + (w * width), (int) (255 * fm[w][h].get(c++)));
						}
					}
				}
			}
		}

		try {
			ImageIO.write(bi, "png", new File(filename));

		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			System.out.println("End: " + filename);
		}

	}
}