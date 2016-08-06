package rd.data;

public class ClassificationDataGenerator {
	/**
	 * Different shape types for generated data
	 * @author azahar
	 *
	 */
	public static enum Shape {
		Square, Random, Curve
	};

	/**
	 * Generate Classifier Data (clusters)
	 * 
	 * @param clusterSpread
	 *            - spread of the 'c' clusters - length = 'c' = number of
	 *            classes
	 * @param centrePoints
	 *            - centre points of the 'c' clusters - centrePoints[x][y] where
	 *            x = 'c' and y = dimensions of input
	 * @param sizePerClass
	 *            - number of points to generate PER class
	 * @param flatten
	 *            - flatter output or not (flatten uses one-hot encoding)
	 * @return
	 */
	public static DataStreamer generate(float clusterSpread[], float[][] centrePoints, int[] sizePerClass,
			boolean flatten, Shape shape) {
		if (clusterSpread.length != centrePoints.length) {
			throw new IllegalArgumentException("The standard deviation array length must match the number of centres");
		}
		final int CENTRES = clusterSpread.length;
		DataStreamer ds = flatten ? new DataStreamer(centrePoints[0].length, CENTRES)
				: new DataStreamer(centrePoints[0].length, 1);
		for (int centre = 0; centre < CENTRES; centre++) {
			float[] centrePoint = centrePoints[centre];
			float spread = clusterSpread[centre] / 2f;
			for (int item = 0; item < sizePerClass[centre]; item++) {
				float input[] = new float[centrePoint.length];
				for (int i = 0; i < centrePoint.length; i++) {

					input[i] = shape(shape, centrePoint[i], spread);

				}

				if (flatten) {
					float[] oneHot = new float[CENTRES];

					oneHot[centre] = 1;
					ds.add(input, oneHot);
				} else {
					ds.add(input, centre);
				}
			}

		}

		return ds;

	}
/**
 * Give the sample points some shape
 * @param shape  - shape type
 * @param centrePoint - centre point
 * @param spread - spread of data
 * @return
 */
	private static float shape(Shape shape, float centrePoint, float spread) {
		switch (shape) {
		case Square:
			if (Math.random() > 0.5) {
				return (float) (centrePoint + (spread * Math.random()));
			} else {
				return (float) (centrePoint - (spread * Math.random()));
			}

		case Random:
			if (Math.random() > 0.5) {
				return (float) (centrePoint * Math.random() * Math.sin(-spread * Math.random()));
			} else {
				return (float) (centrePoint * Math.random() * Math.cos(-spread * Math.random()));
			}

		case Curve:
			if (Math.random() > 0.5) {
				return (float) ((centrePoint) + Math.exp(-Math.random() * spread));
			} else {
				return (float) ((centrePoint) - Math.exp(-Math.random() * spread));
			}
		default:
			return (float) Math.random();
		}
	}
}
