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

	//Filename to save/load the model
	private static final String fileName = "network2.out.nw";
	//Load from file flag - needs to be set and the filename variable needs to point to the right file
	private static final boolean loadFromFile = true;

	//Random generator
	private static final Random rnd = new Random(123);
	
	/**
	 * Files for Labels/Images - Training and Testing - change to run tests
	 */
	private static final String D_ML_STATS_MNIST_TRAIN_LABELS = "data\\train-labels.idx1-ubyte";
	private static final String D_ML_STATS_MNIST_TRAIN_IMAGES = "data\\train-images.idx3-ubyte";
	private static final String D_ML_STATS_MNIST_T10K_LABELS = "data\\t10k-labels.idx1-ubyte";
	private static final String D_ML_STATS_MNIST_T10K_IMAGES = "data\\t10k-images.idx3-ubyte";

	public static void main(String... args) throws FileNotFoundException, IOException {

		//Threaded Image Writer - writer for example images with  4 threads.
		ThreadedImageWriter tiw = new ThreadedImageWriter(4);
		
		//Epochs for CD
		int epoch = 10000;

		//Load Test and Training data
		DataStreamer train = MnistToDataStreamer.createStreamer(D_ML_STATS_MNIST_T10K_IMAGES,
				D_ML_STATS_MNIST_T10K_LABELS);
		System.out.println("Training Streamer Ready!");

		DataStreamer test = MnistToDataStreamer.createStreamer(D_ML_STATS_MNIST_TRAIN_IMAGES,
				D_ML_STATS_MNIST_TRAIN_LABELS);
		System.out.println("Testing Streamer Ready!");

		//Our network - consists of a list of layers
		List<LayerIf> network;
		
		//width of hidden units - constant value
		int width = 10;
		//Number of hidden units - square of the width variable
		int lengthHidden = width * width;
		
		//Check if we want to load a model from a file or create a new one and save it to the file
		if (!(new File(fileName)).exists() || !loadFromFile) {

			//Recipe for our network: >> 784 - 100 - 100 - 100 >>
			String recipe = "STOCHASTIC 784 " + lengthHidden + "\nSTOCHASTIC " + lengthHidden + " 100\nSTOCHASTIC "
					+ lengthHidden + " " + lengthHidden + "\nSTOCHASTIC " + lengthHidden + " " + lengthHidden;

			network = RecipeNetworkBuilder.build(recipe);
			
			StochasticNetwork nw = new StochasticNetwork(network, Function.LOGISTIC, Function.LOGISTIC);
			List<FloatMatrix> miniBatch = new ArrayList<>();
			for (int i = 0; i < epoch; i++) {
				miniBatch.clear();
				for (FloatMatrix input : train) {
					//Randomly add elements to the mini batch
					if (Math.random() < 0.001) {
						miniBatch.add(input);
					}
				}
				
				//Pre-train CD-10
				nw.preTrain(miniBatch, 10, 0.02f);

				if (epoch % 100 == 0) {
					System.out.println(i * 100f / epoch);
				}
			}

			//Save the trained network
			StochasticNetwork.save(fileName, network);
		} else {
			//Load the network from filename provided
			network = StochasticNetwork.load(fileName);
		}
		
		
		float avg = 0;
		int count = 0;
		
		//TEST
		
		//Max instances to record in images
		int instancesRemainingToRecord = 20;
		
		//Should be equal to number of hidden layers
		int recordLen = 4;
		
		FloatMatrix combinedInstances[][] = new FloatMatrix[instancesRemainingToRecord][recordLen + 2];
		FloatMatrix[] hiddenLayerRecords = new FloatMatrix[recordLen];
		for (FloatMatrix input : test) {

			FloatMatrix h = input;
			int lcount = 0;
			for (LayerIf l : network) {
				h = StochasticLayer.stochasticLayer(Propagate.upOne(h, l), rnd);
				hiddenLayerRecords[lcount++] = h;
			}

			FloatMatrix v = StochasticLayer.stochasticLayer(Propagate.down(h, network), rnd);

			avg += PatternBuilder.score(v, input, 0.1f);
			if (Math.random() < 0.01 && instancesRemainingToRecord > 0) {
				instancesRemainingToRecord--;
				combinedInstances[instancesRemainingToRecord][0] = input;
				combinedInstances[instancesRemainingToRecord][1] = v;
				for (int cc = 0; cc < hiddenLayerRecords.length; cc++) {
					combinedInstances[instancesRemainingToRecord][cc + 2] = hiddenLayerRecords[cc];
				}

			}
			count++;
		}

		//Write the combined image of inputs, outputs and hidden layer activations while testing
		tiw.writeImage(combinedInstances, 28, 28, "combine.png");

		
		//Generating with Random Feature inputs
		
		int rh = 0, rw = 0;

		int maxH = 20, maxW = 20;
		
		//Maximum number of digits to be generated.
		int maxDigitsToBeGenerated = maxH * maxW;
		
		//Maximum Sample steps
		int maxSample= 50;
		
		FloatMatrix[][] randomGen = new FloatMatrix[maxH][maxW];
		for (int i = 0; i < maxDigitsToBeGenerated; i++) {
			FloatMatrix randFeatureSet = FloatMatrix.rand(lengthHidden, 1);
			for (int j = 0; j < lengthHidden; j++) {
				if (Math.random() > 0.5) {
					randFeatureSet.put(j, 0, 1f);
				} else {
					randFeatureSet.put(j, 0, 0f);
				}
			}

			//Do up down with hidden clampled to random value
			FloatMatrix random = randFeatureSet;
			
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

		//Write Generated Digits
		tiw.writeImage(randomGen, 28, 28, "random"+(maxSample+1)+"."+maxDigitsToBeGenerated+".png");

		System.out.println(avg / count);
		tiw.shutdown();

	}

}

/**
 * Threaded Image Writer
 * @author azahar
 *
 */
class ThreadedImageWriter {
	private final ExecutorService es;

	/**
	 * Number of threads
	 * @param threads
	 */
	public ThreadedImageWriter(int threads) {
		es = Executors.newFixedThreadPool(threads);
	}

	/**
	 * Write image with single float matrix
	 * @param fm 
	 * @param width
	 * @param height
	 * @param filename
	 */
	public void writeImage(FloatMatrix fm, int width, int height, String filename) {

		es.execute(new Task(fm, width, height, filename));
	}

	/**
	 * Write a set of float matrixs into a grouped image
	 * @param fm - matrix of float matrix
	 * @param width
	 * @param height
	 * @param filename
	 */
	public void writeImage(FloatMatrix fm[][], int width, int height, String filename) {

		es.execute(new Task(fm, width, height, filename));
	}

	public void shutdown() {
		es.shutdown();
	}

}

/**
 * Task to write float matrix to image (png)
 * @author azahar
 *
 */
class Task implements Runnable {
	private final FloatMatrix fm[][];
	private final int width, height;
	private final String filename;

	/**
	 * 
	 * @param fm - float matrix
	 * @param width - width of image
	 * @param height - height of image
	 * @param filename - filename
	 */
	public Task(FloatMatrix fm, int width, int height, String filename) {
		this.fm = new FloatMatrix[][] { { fm } };
		this.width = width;
		this.height = height;
		this.filename = filename;
	}

	/**
	 * 
	 * @param fm - matrix of float matrix to draw grouped image.
	 * @param width - width of image
	 * @param height - height of image
	 * @param filename - filename
	 */
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