package rd.neuron.neuron.test;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

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

/**
 * Descriminative Test
 * 
 * @author azahar
 *
 */
public class TestRBMMNISTRecipeClassifier {

	// Filename to save/load the model
	private static final String fileName = "network.discrm.16.reduced.nw";
	// Load from file flag - needs to be set and the filename variable needs to
	// point to the right file
	private static final boolean loadFromFile = false;

	// Random generator
	private static final Random rnd = new Random();

	/**
	 * Files for Labels/Images - Training and Testing - change to run tests
	 */
	private static final String D_ML_STATS_MNIST_TRAIN_LABELS = "data\\train-labels.idx1-ubyte";
	private static final String D_ML_STATS_MNIST_TRAIN_IMAGES = "data\\train-images.idx3-ubyte";
	private static final String D_ML_STATS_MNIST_T10K_LABELS = "data\\t10k-labels.idx1-ubyte";
	private static final String D_ML_STATS_MNIST_T10K_IMAGES = "data\\t10k-images.idx3-ubyte";

	public static void main(String... args) throws FileNotFoundException, IOException {

		// Number of output layers
		int numOutputLayers = 1;
		// Threaded Image Writer - writer for example images with 4 threads.
		ThreadedImageWriter tiw = new ThreadedImageWriter(4);

		// Epochs for CD
		int epoch = 5000;

		// Load Test and Training data
		DataStreamer test = MnistToDataStreamer.createStreamer(D_ML_STATS_MNIST_T10K_IMAGES,
				D_ML_STATS_MNIST_T10K_LABELS);
		System.out.println("Training Streamer Ready!");

		DataStreamer train = MnistToDataStreamer.createStreamer(D_ML_STATS_MNIST_TRAIN_IMAGES,
				D_ML_STATS_MNIST_TRAIN_LABELS);
		System.out.println("Testing Streamer Ready!");

		// Our network - consists of a list of layers - stochastic network is
		// the part that will be pretrained.
		List<LayerIf> network, stochasticNetwork;

		//Number of instances to use for Fine Tuning
		int ftTrainInstanceCount = 40000;
		
		// Mini batch size
		int miniBatchSize = 100;

		// width of hidden units - constant value
		int width = 16;
		// Number of hidden units - square of the width variable
		int lengthHidden = width * width;

		// Check if we want to load a model from a file or create a new one and
		// save it to the file
		if (!(new File(fileName)).exists() || !loadFromFile) {

			// Recipe for our network: >> 784 - lengthHidden - lengthHidden - lengthHidden - output >>
			String recipe = "STOCHASTIC 784 " + lengthHidden + "\nSTOCHASTIC " + lengthHidden + " " + lengthHidden
					+ "\nSTOCHASTIC " + lengthHidden + " " + lengthHidden + "\nSTOCHASTIC " + lengthHidden + " "
					+ lengthHidden + "\nRANDOM " + lengthHidden + " 10";
			


			network = RecipeNetworkBuilder.build(recipe);
			stochasticNetwork = getStochasticNetwork(network, numOutputLayers);

			List<FloatMatrix> miniBatch = new ArrayList<>(miniBatchSize);
			for (int i = 0; i < epoch; i++) {
				miniBatch.clear();
				for (int b = 0; b < miniBatchSize; b++) {
					// Randomly add elements to the mini batch
					miniBatch.add(train.getRandom());
				}

				// Pre-train CD-10
				StochasticNetwork.preTrain(stochasticNetwork, miniBatch, 10, 0.02f);

				if (epoch % 100 == 0) {
					System.out.println(i * 100f / epoch);
				}
			}
			StochasticNetwork sn = new StochasticNetwork(network, Function.LOGISTIC, Function.LOGISTIC);

			// Fine Tuning
			System.out.println("Fine Tuning");
			
		

			while (ftTrainInstanceCount > 0) {
				FloatMatrix ft = train.getRandom();
				sn.fineTuneOutputLayer(train.getOutput(ft), Propagate.up(ft, network));
				ftTrainInstanceCount--;
			}

			// Save the trained network
			StochasticNetwork.save(fileName, network);
		} else {
			// Load the network from filename provided
			network = StochasticNetwork.load(fileName);
			stochasticNetwork = getStochasticNetwork(network, numOutputLayers);
		}

		float avg = 0;
		float avgMatchScore = 0;
		int count = 0;

		// TEST

		System.out.println("Network size: " + network.size());

		System.out.println("Stochastic Network size: " + stochasticNetwork.size());

		// Max instances to record in images
		int instancesRemainingToRecord = 20;

		// Should be equal to number of hidden layers
		int recordLen = 4;

		int errorCount = 0;
		FloatMatrix combinedInstances[][] = new FloatMatrix[instancesRemainingToRecord][recordLen + 1];
		FloatMatrix[] hiddenLayerRecords = new FloatMatrix[recordLen];
		for (FloatMatrix input : test) {

			FloatMatrix h = input;
			int lcount = 0;
			for (LayerIf l : stochasticNetwork) {
				h = StochasticLayer.stochasticLayer(Propagate.upOne(h, l), rnd);
				hiddenLayerRecords[lcount++] = h;
			}

			FloatMatrix testOut = Propagate.up(input, network);
			FloatMatrix expectedOut = test.getOutput(input);
			float score = PatternBuilder.score(testOut, expectedOut, 0.1f);
			float matchScore = PatternBuilder.matchScore(testOut, expectedOut);

			if (matchScore == 0) {
				tiw.writeImage(input, 28, 28, "error\\error_" + (++errorCount) + ".png");
				System.out.println(
						errorCount + ": " + testOut + "\n  > " + expectedOut + "\n" + matchScore + "\n" + score);
			}
			avg += score;
			avgMatchScore += matchScore;
			if (Math.random() < 0.01 && instancesRemainingToRecord > 0) {
				instancesRemainingToRecord--;
				combinedInstances[instancesRemainingToRecord][0] = input;
				for (int cc = 0; cc < hiddenLayerRecords.length; cc++) {
					combinedInstances[instancesRemainingToRecord][cc + 1] = hiddenLayerRecords[cc];
				}

			}
			count++;
		}
		if (instancesRemainingToRecord > 0) {
			System.out.println("Problem, not all instances are initialised: " + instancesRemainingToRecord);
		}
		// Write the combined image of inputs, outputs and hidden layer
		// activations while testing
		tiw.writeImage(combinedInstances, 28, 28, "combine.png");

		System.out.println("Closeness Average: " + avg / count);
		System.out.println("Peak Match: " + avgMatchScore / count);

		// Generating with Random Feature inputs

		int rh = 0, rw = 0;

		int maxH = 20, maxW = 20;

		// Maximum number of digits to be generated.
		int maxDigitsToBeGenerated = maxH * maxW;

		// Maximum Sample steps
		int maxSample = 50;

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

			// Do up down with hidden clampled to random value
			FloatMatrix random = randFeatureSet;

			for (int cc = 0; cc < maxSample; cc++) {
				random = StochasticLayer.stochasticLayer(Propagate.down(random, stochasticNetwork), rnd);

				random = StochasticLayer.stochasticLayer(Propagate.up(random, stochasticNetwork), rnd);

			}

			random = StochasticLayer.stochasticLayer(Propagate.down(random, stochasticNetwork), rnd);

			randomGen[rw++][rh] = random;
			if (rh >= maxH) {
				rh = 0;
			}
			if (rw >= maxW) {
				rw = 0;
				rh++;
			}

		}

		// Write Generated Digits
		tiw.writeImage(randomGen, 28, 28, "random" + (maxSample + 1) + "." + maxDigitsToBeGenerated + ".png");

	

		tiw.shutdown();

	}

	public static List<LayerIf> getStochasticNetwork(List<LayerIf> network, int numOutputLayers) {
		if (numOutputLayers == 0) {
			return network;
		} else {
			return network.subList(0, network.size() - numOutputLayers);
		}
	}

}
