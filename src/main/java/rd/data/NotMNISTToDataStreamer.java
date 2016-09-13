package rd.data;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.StreamSupport;

import javax.imageio.ImageIO;

import org.apache.log4j.Logger;
/**
 * NotMNIST to Data Streamer.
 * Takes in root directory of standard MNIST dataset which is organised as:
 * <folder-label>/<files - examples>
 * So to get the label information for an image we have to look at its parent folder.
 * Some images do not parse well so we need to reject them ... usually the number is insignificant (5 out of roughly 600k images)
 * @author azahar
 *
 */
public class NotMNISTToDataStreamer {

	private static final int IMAGE_SIZE_PIXELS = 784;
	private static final Logger logger = Logger.getLogger(NotMNISTToDataStreamer.class);
	private static final int MAX_ALPHABETS = 26;
	private static final float oneHotAlphabet[][] = buildCharToOneHot();

	/**
	 * Get one hot encoding for letter of the English alphabet
	 * @param c - can be upper or lower case
	 * @return
	 */
	public static float[] getOneHotEncoding(char c) {
		c = Character.toUpperCase(c);
		int index = c - 'A';
		if (index >= oneHotAlphabet.length) {
			return oneHotAlphabet[MAX_ALPHABETS];
		}

		return oneHotAlphabet[index];
	}

	/**
	 * Build one-hot encoding labels
	 * @return
	 */
	private static float[][] buildCharToOneHot() {

		float[][] temp = new float[MAX_ALPHABETS + 1][MAX_ALPHABETS];

		for (char lbl = 'A'; lbl <= 'Z'; lbl++) {

			int index = lbl - 'A';

			for (int i = 0; i < MAX_ALPHABETS; i++) {

				temp[index][i] = i == index ? 1 : 0;
			}

		}
		for (int i = 0; i < MAX_ALPHABETS; i++) {
			temp[26][i] = 0;
		}

		return temp;
	}

	/**
	 * 
	 * @param rootDir
	 *            - for the NotMnist data set
	 * @param featureLength
	 *            - length of feature input vector = height * width; must be
	 *            same for all inputs
	 * @param parallelInstances
	 *            - number of parallel instances to use for processing images
	 * @return
	 */
	public static DataStreamer createStreamer(String rootDir, int featureLength, int parallelInstances) {

		DataStreamer ds = new DataStreamer(featureLength, MAX_ALPHABETS);
		List<File> allFiles = new ArrayList<>(600000);
		long st = System.currentTimeMillis();
		DirectoryParser.getFiles(allFiles, new File(rootDir), DirectoryParser.IMG_FILE_FILTER, false);
		System.out.println(((System.currentTimeMillis() - st) / 1000) + "(secs) for " + allFiles.size());
		st = System.currentTimeMillis();
		final AtomicInteger processCount = new AtomicInteger(0);
		final AtomicInteger error = new AtomicInteger(0);
		final AtomicInteger added = new AtomicInteger(0);
		ForkJoinPool fjp = new ForkJoinPool(parallelInstances);
		try {
			fjp.submit(new Runnable() {
				@Override
				public void run() {

					StreamSupport.stream(allFiles.spliterator(), true).parallel().map(f -> {

						try {

							BufferedImage buff = ImageIO.read(f);
							if (buff == null) {
								logger.error("Error: Null Image " + f.getPath());
								return f;
							}
							int w = buff.getWidth(), h = buff.getHeight();
							float flat[] = new float[h * w];
							if (flat.length == featureLength) {
								int i = 0;
								for (int x = 0; x < h; x++) {
									for (int y = 0; y < w; y++) {
										flat[i++] = buff.getRGB(x, y) / 255f;
									}
								}
								char labelChar = f.getParentFile().getName().trim().charAt(0);

								float[] label = getOneHotEncoding(labelChar);
								ds.add(flat, label);
								added.incrementAndGet();
							} else {
								error.incrementAndGet();
								System.err.println(
										"Bad dimension of image: " + f.getPath() + "  Dimension: " + h + "x" + w);
							}

							processCount.incrementAndGet();
						} catch (IOException e) {
							logger.error("Not able to read file: " + f.getName() + "\nError:" + e);
							e.printStackTrace(System.err);
							error.incrementAndGet();
						}

						return f;
					}).count();
				}
			}).get();
			if (logger.isInfoEnabled()) {
				logger.info("Processed: " + processCount.get() + " added: " + added.get() + " in "
						+ ((System.currentTimeMillis() - st) / 1000) + "(secs) for (%)"
						+ (processCount.get() * 100f / allFiles.size()) + "\nError (%): "
						+ (error.get() * 100f / allFiles.size()));
			}

		} catch (InterruptedException e) {

			e.printStackTrace();
		} catch (ExecutionException e) {

			e.printStackTrace();
		}

		return ds;
	}

	/**
	 * Test Implementation 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String... args) throws Exception {

		int parallelInstances = 8;
		DataStreamer ds = createStreamer("d:\\ml stats\\notMnist\\notMnist_large", IMAGE_SIZE_PIXELS,parallelInstances);
		logger.info("Output: "+StreamSupport.stream(ds.spliterator(), false).count());

	}

}
