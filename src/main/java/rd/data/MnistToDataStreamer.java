package rd.data;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.canova.image.mnist.MnistLabelFile;

/**
 * Convert MNIST Image/Label pairs to Data Streamer
 * 
 * @author azahar
 *
 */
public class MnistToDataStreamer {

	private static final float MAX_SAT = 255f;
	private static final int OUTPUT_LENGTH = 10;
	private static final int INPUT_LENGTH = 784;

	/**
	 * Uses default values for max saturation = 255, INPUT Length = 784 and
	 * OUTPUT Length = 10
	 * 
	 * @param imgFile
	 * @param lblFile
	 * @return
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static DataStreamer createStreamer(String imgFile, String lblFile)
			throws FileNotFoundException, IOException {
		return createStreamer(imgFile, lblFile, MAX_SAT, INPUT_LENGTH);
	}

	/**
	 * 
	 * @param imgFile
	 * @param lblFile
	 * @param maxValue
	 *            = of saturation to normalise values between 0 and 1 (set to 1
	 *            to not normalise)
	 * @param featureLength
	 *            = length of feature vector (usually flattened to 784)
	 * @return
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static DataStreamer createStreamer(String imgFile, String lblFile, float maxValue, int featureLength)
			throws FileNotFoundException, IOException {
		DataStreamer streamer = new DataStreamer(featureLength, OUTPUT_LENGTH);

		MnistImageFile digitImage = new MnistImageFile(imgFile, "r");
		MnistLabelFile labelData = new MnistLabelFile(lblFile, "r");

		final float max = maxValue;

		while (digitImage.getCurrentIndex() <= digitImage.getCount()) {

			int lbl = labelData.readLabel();

			int img[][] = digitImage.readImage();

			float[] imgInput = new float[img.length * img[0].length];
			int k = 0;

			for (int i = 0; i < img.length; i++) {
				for (int j = 0; j < img[0].length; j++) {

					imgInput[k++] = (float) (img[i][j] / max);
				}

			}

			streamer.add(imgInput, digitToOneHotEncoding(lbl));

		}

		digitImage.close();
		labelData.close();
		return streamer;
	}

	private static float[] zero = null, one = null, two = null, three = null, four = null, five = null, six = null,
			seven = null, eight = null, nine = null, def = null;

	public static float[] digitToOneHotEncoding(int digit) {

		switch (digit) {
		case 0:
			if (zero != null) {
				return zero;
			}
			return zero = new float[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		case 1:
			if (one != null) {
				return one;
			}
			return one = new float[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
		case 2:
			if (two != null) {
				return two;
			}
			return two = new float[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
		case 3:
			if (three != null) {
				return three;
			}
			return three = new float[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
		case 4:
			if (four != null) {
				return four;
			}
			return four = new float[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
		case 5:
			if (five != null) {
				return five;
			}
			return five = new float[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
		case 6:
			if (six != null) {
				return six;
			}
			return six = new float[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
		case 7:
			if (seven != null) {
				return seven;
			}
			return seven = new float[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };
		case 8:
			if (eight != null) {
				return eight;
			}
			return eight = new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
		case 9:
			if (nine != null) {
				return nine;
			}
			return nine = new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
		default:
			if (def != null) {
				return def;
			}
			return def = new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		}
	}

}
